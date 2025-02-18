# Image Show

This project compares the speed of video rendering between Python and C++.

## Rendering time

- Desktop(pypi)

| Language | Library         |  Time(msec)   |
| :------: | :-------------- | :-----------: |
|  Python  | OpenCV(4.10.0)  | 2.61 ± 0.0520 |
|  Python  | PyOpenGL(3.1.7) | 1.33 ± 0.0776 |
|   C++    | OpenCV(4.5.4)   | 1.29 ± 0.1949 |

- Desktop(build)

| Language | Library        |  Time(msec)   |
| :------: | :------------- | :-----------: |
|  Python  | OpenCV(4.10.0) | 1.26 ± 0.1894 |
|   C++    | OpenCV(4.10.0) | 1.22 ± 0.1433 |

- Kria KV260(pypi)

| Language | Library         |  Time(msec)   |
| :------: | :-------------- | :-----------: |
|  Python  | OpenCV(4.10.0)  | 8.70 ± 0.0864 |
|  Python  | PyOpenGL(3.1.7) | 58.9 ± 1.5048 |
|   C++    | OpenCV(4.6.0)   | 9.52 ± 0.0894 |

## Setup

### python

```
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ python -c "import cv2; print(cv2.getBuildInformation())"
General configuration for OpenCV 4.10.0 =====================================
  Version control:               4.10.0-dirty

  Platform:
    Timestamp:                   2024-06-17T17:56:37Z
    Host:                        Linux 5.15.0-1064-azure x86_64
    CMake:                       3.29.5
    CMake generator:             Unix Makefiles
    CMake build tool:            /bin/gmake
    Configuration:               Release

  CPU/HW features:
    Baseline:                    SSE SSE2 SSE3
      requested:                 SSE3
    Dispatched code generation:  SSE4_1 SSE4_2 FP16 AVX AVX2 AVX512_SKX
      requested:                 SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX
      SSE4_1 (16 files):         + SSSE3 SSE4_1
      SSE4_2 (1 files):          + SSSE3 SSE4_1 POPCNT SSE4_2
      FP16 (0 files):            + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 AVX
      AVX (8 files):             + SSSE3 SSE4_1 POPCNT SSE4_2 AVX
      AVX2 (36 files):           + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 FMA3 AVX AVX2
      AVX512_SKX (5 files):      + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 FMA3 AVX AVX2 AVX_512F AVX512_COMMON AVX512_SKX

  C/C++:
    Built as dynamic libs?:      NO
    C++ standard:                11
    C++ Compiler:                /opt/rh/devtoolset-10/root/usr/bin/c++  (ver 10.2.1)
    C++ flags (Release):         -Wl,-strip-all   -fsigned-char -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -O3 -DNDEBUG  -DNDEBUG
    C++ flags (Debug):           -Wl,-strip-all   -fsigned-char -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -g  -O0 -DDEBUG -D_DEBUG
    C Compiler:                  /opt/rh/devtoolset-10/root/usr/bin/cc
    C flags (Release):           -Wl,-strip-all   -fsigned-char -W -Wall -Wreturn-type -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -O3 -DNDEBUG  -DNDEBUG
    C flags (Debug):             -Wl,-strip-all   -fsigned-char -W -Wall -Wreturn-type -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -g  -O0 -DDEBUG -D_DEBUG
    Linker flags (Release):      -Wl,--exclude-libs,libippicv.a -Wl,--exclude-libs,libippiw.a -L/ffmpeg_build/lib  -Wl,--gc-sections -Wl,--as-needed -Wl,--no-undefined  
    Linker flags (Debug):        -Wl,--exclude-libs,libippicv.a -Wl,--exclude-libs,libippiw.a -L/ffmpeg_build/lib  -Wl,--gc-sections -Wl,--as-needed -Wl,--no-undefined  
    ccache:                      YES
    Precompiled headers:         NO
    Extra dependencies:          /lib64/libopenblas.so Qt5::Core Qt5::Gui Qt5::Widgets Qt5::Test Qt5::Concurrent /usr/local/lib/libpng.so /lib64/libz.so dl m pthread rt
    3rdparty dependencies:       libprotobuf ade ittnotify libjpeg-turbo libwebp libtiff libopenjp2 IlmImf ippiw ippicv

  OpenCV modules:
    To be built:                 calib3d core dnn features2d flann gapi highgui imgcodecs imgproc ml objdetect photo python3 stitching video videoio
    Disabled:                    world
    Disabled by dependency:      -
    Unavailable:                 java python2 ts
    Applications:                -
    Documentation:               NO
    Non-free algorithms:         NO

  GUI:                           QT5
    QT:                          YES (ver 5.15.13 )
      QT OpenGL support:         NO
    GTK+:                        NO
    VTK support:                 NO

  Media I/O: 
    ZLib:                        /lib64/libz.so (ver 1.2.7)
    JPEG:                        build-libjpeg-turbo (ver 3.0.3-70)
      SIMD Support Request:      YES
      SIMD Support:              YES
    WEBP:                        build (ver encoder: 0x020f)
    PNG:                         /usr/local/lib/libpng.so (ver 1.6.43)
    TIFF:                        build (ver 42 - 4.6.0)
    JPEG 2000:                   build (ver 2.5.0)
    OpenEXR:                     build (ver 2.3.0)
    HDR:                         YES
    SUNRASTER:                   YES
    PXM:                         YES
    PFM:                         YES

  Video I/O:
    DC1394:                      NO
    FFMPEG:                      YES
      avcodec:                   YES (59.37.100)
      avformat:                  YES (59.27.100)
      avutil:                    YES (57.28.100)
      swscale:                   YES (6.7.100)
      avresample:                NO
    GStreamer:                   NO
    v4l/v4l2:                    YES (linux/videodev2.h)

  Parallel framework:            pthreads

  Trace:                         YES (with Intel ITT)

  Other third-party libraries:
    Intel IPP:                   2021.11.0 [2021.11.0]
           at:                   /io/_skbuild/linux-x86_64-3.9/cmake-build/3rdparty/ippicv/ippicv_lnx/icv
    Intel IPP IW:                sources (2021.11.0)
              at:                /io/_skbuild/linux-x86_64-3.9/cmake-build/3rdparty/ippicv/ippicv_lnx/iw
    VA:                          NO
    Lapack:                      YES (/lib64/libopenblas.so)
    Eigen:                       NO
    Custom HAL:                  NO
    Protobuf:                    build (3.19.1)
    Flatbuffers:                 builtin/3rdparty (23.5.9)

  OpenCL:                        YES (no extra features)
    Include path:                /io/opencv/3rdparty/include/opencl/1.2
    Link libraries:              Dynamic load

  Python 3:
    Interpreter:                 /opt/python/cp39-cp39/bin/python3.9 (ver 3.9.19)
    Libraries:                   libpython3.9m.a (ver 3.9.19)
    Limited API:                 YES (ver 0x03060000)
    numpy:                       /home/ci/.local/lib/python3.9/site-packages/numpy/_core/include (ver 2.0.0)
    install path:                python/cv2/python-3

  Python (for build):            /opt/python/cp39-cp39/bin/python3.9

  Java:                          
    ant:                         NO
    Java:                        NO
    JNI:                         NO
    Java wrappers:               NO
    Java tests:                  NO

  Install to:                    /io/_skbuild/linux-x86_64-3.9/cmake-install
-----------------------------------------------------------------
```

### C++

```
$ sudo apt install cmake libopencv-dev libfmt-dev build-essential 
$ opencv_version
4.5.4
```

```
$ mkdir cpp/build
$ cd cpp/build
$ cmake ..
$ make
```

