# DVS Reconstruction

## Installation Prerequisites

Before you compile everything as explained below, you should make the following
exports with the values adapted as appropriate.

```sh
export COMPUTE_CAPABILITY=5.2
export CUDA_SDK_ROOT_DIR=/opt/cuda/samples/
export IMAGEUTILITIES_ROOT=$(realpath ./imageutilities)
```

## Actual Readme

This repository provides software for our publication "Real-Time Intensity-Image Reconstruction for Event Cameras Using Manifold Regularisation", BMVC 2016.

If you use this code please cite the following publication:
~~~
@inproceedings{reinbacher_bmvc2016,
  author = {Christian Reinbacher and Gottfried Graber and Thomas Pock},
  title = {{Real-Time Intensity-Image Reconstruction for Event Cameras Using Manifold Regularisation}},
  booktitle = {2016 British Machine Vision Conference (BMVC)},
  year = {2016},
}
~~~

## Compiling
For your convenience, the required libraries that are on Github are added as
submodules. So clone this repository with `--recursive` or do a
~~~
git submodule update --init --recursive
~~~
after cloning.

This software requies:
 - GCC >= 4.9
 - CMake >= 3.2
 - Qt >= 5.6
 - ImageUtilities (https://github.com/VLOGroup/imageutilities) with the `iugui`, `iuio` and `iumath` modules
 - libcaer >=2.0 (https://github.com/inilabs/libcaer)
 - cnpy (https://github.com/rogersce/cnpy)
 - DVS128 or DAVIS240 camera (can also load events from files)

To compile, first build and install ImageUtilities, then:
 ~~~
cd cnpy
cmake .
make
(sudo) make install
cd ../libcaer
cmake .
make
(sudo) make install
cd ..
mkdir build
cd build
cmake ../src
make -j6
 ~~~

 Per default, the application will compile to support the iniLabs DVS128. If you want to attach a DAVIS240 instead, set the CMake option `WITH_DAVIS`.

## Usage
Launch `live_reconstruction_gui` to get to the main application which should look like this:
<img src="https://github.com/VLOGroup/dvs-reconstruction/raw/master/images/screenshot.png"></img>
Clicking on the play button with an attached camera will start the live reconstruction method. Alternatively, events can be loaded from text files with one event per line:
~~~
<timestamp in seconds> <x> <y> <polarity (-1/1)>
...
...
~~~
