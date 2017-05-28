#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <tclap/CmdLine.h>

#include "common.cuh"
#include "common.h"
#include "csv.h"
#include "denoise.h"
#include "iu/iucore.h"
#include "iu/iuio.h"
#include "iu/iumath.h"

struct SaneEvent {
  std::uint64_t t;
  std::uint16_t x;
  std::uint16_t y;
  std::int8_t p;
};

int main(int argc, char **argv) {
  TCLAP::CmdLine cmd("Reconstruct intensity images from events");
  TCLAP::ValueArg<std::uint64_t> frame_interval_arg(
      "f", "frame-interval", "Reconstruct a frame every n milliseconds", false,
      1000, "ms");
  TCLAP::ValueArg<int> width_arg("W", "width", "Image width", false, 128, "n");
  TCLAP::ValueArg<int> height_arg("H", "height", "Image height", false, 128,
                                  "n");
  TCLAP::UnlabeledValueArg<std::string> events_arg(
      "events-path", "Path to events.csv", true, "", "path");
  TCLAP::UnlabeledValueArg<std::string> out_arg(
      "out-path", "Directory to store the frames into", true, "", "path");

  cmd.add(frame_interval_arg);
  cmd.add(width_arg);
  cmd.add(height_arg);
  cmd.add(events_arg);
  cmd.add(out_arg);

  try {
    cmd.parse(argc, argv);
  } catch (TCLAP::ArgException &e) {
    std::cout << e.error() << std::endl;
    return 1;
  }

  const auto frame_interval = frame_interval_arg.getValue();
  const auto width = width_arg.getValue();
  const auto height = height_arg.getValue();
  const auto events_path = events_arg.getValue();
  const auto out_path = out_arg.getValue();

  std::vector<struct SaneEvent> events;
  try {
    io::CSVReader<4> reader(events_path);
    reader.read_header(io::ignore_extra_column, "timestamp", "x", "y",
                       "polarity");

    std::uint64_t timestamp;
    std::uint16_t x, y;
    std::int8_t polarity;
    while (reader.read_row(timestamp, x, y, polarity)) {
      struct SaneEvent ev;
      ev.t = timestamp;
      ev.x = x;
      // Mirror images on the y-axis
      ev.y = height - y - 1;
      ev.p = polarity > 0 ? 1 : -1;
      events.push_back(ev);
    }
  } catch (io::error::base &e) {
    std::cout << e.what() << std::endl;
    return 1;
  }

  // Denoising parameters
  const double u0 = 1.5;
  const double u_min = 1.0;
  const double u_max = 2.0;
  const double lambda = 90.0;
  const double lambda_t = 2.0;
  const double C1 = 1.15;
  const double C2 = 1.25;
  const auto method = TV_LogL2;
  const int iterations = 50;

  // Denoising state
  auto output = new iu::ImageGpu_32f_C1(width, height);
  auto input = new iu::ImageGpu_32f_C1(width, height);
  auto old_timestamp = new iu::ImageGpu_32f_C1(width, height);

  // Initialize GPU state
  iu::math::fill(*output, u0);
  iu::copy(output, input);
  cuda::initDenoise(output, old_timestamp);
  iu::math::fill(*old_timestamp, 0.0);

  auto frame_end = events[0].t - events[0].t % frame_interval + frame_interval;
  std::vector<struct SaneEvent> buffer;
  for (auto &e : events) {
    if (e.t > frame_end) {
      iu::LinearHostMemory_32f_C4 events_host(buffer.size());
      for (std::size_t i = 0; i < buffer.size(); ++i) {
        const auto ev = buffer[i];
        *events_host.data(i) =
            make_float4(ev.x, ev.y, ev.p, ev.t * TIME_CONSTANT);
      }
      cuda::setEvents(input, old_timestamp, &events_host, C1, C2);

      cuda::solveTVIncrementalManifold(output, input, old_timestamp, lambda,
                                       lambda_t, iterations, u_min, u_max,
                                       method);

      iu::imsave(output, out_path + "/" + std::to_string(frame_end) + ".png", true);

      // Start a new frame
      buffer.clear();
      frame_end += frame_interval;
    }

    buffer.push_back(e);
  }

  return 0;
}
