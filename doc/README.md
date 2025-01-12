# Build OPENCV VERSION 4.10.0 on: Ubuntu 22.04, GPU Driver Version: 550.120, CUDA Version: 11.5, GCC-11

## Python

```
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install wheel numpy tesseract
$ which python
```

sysconfigを使ってCMAKE時のパスを確認する。

```
$ python -c "import sysconfig; print(sysconfig.get_paths()['include'])"
/usr/include/python3.10
$ python -c "
import sysconfig
from pathlib import Path

libdir = Path(sysconfig.get_config_var('LIBDIR'))
ldlibrary = sysconfig.get_config_var('LDLIBRARY')  # 例: 'libpython3.10.so'
print(libdir / ldlibrary)
"
/usr/lib/x86_64-linux-gnu/libpython3.10.so
$ sudo apt install python3.10-dev
```

以下のディレクトリが必要になる。

- `/home/kota/Work/imshow/.venv/bin/python`
- `/home/kota/Work/imshow/.venv/lib/python3.10/site-packages/`
- `/usr/include/python3.10`
- `/usr/lib/x86_64-linux-gnu/libpython3.10.so`

## GPU

NVIDIA Driver, CUDA, cuDNNはaptで管理する。

```
$ sudo apt install cuda-drivers # Recommendを使う
$ sudo apt install cuda-toolkit # Driver versionに合わせてくれる 
$ sudo apt install nvidia-cudnn # 
```

```
$ nvidia-smi --version
$ nvcc --version
$ ls /usr/lib/x86_64-linux-gnu/libcudnn* # インストール内容を確認 
```

CUDA_ARCHは[HERE](https://developer.nvidia.com/cuda-gpus)で確認する。
GeForce RTX 3080 Tiは`8.6`だった。

## Install dependencies and recommeneded packages

下記のパッケージをインストールして依存関係を解決しておく。
[参考にしたページ](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)。

```
$ sudo apt update
$ sudo apt upgrade

# Image I/O libs
$ sudo apt install libjpeg-dev libpng-dev libtiff-dev

# Install basic codec libraries
$ sudo apt install libavcodec-dev libavformat-dev libswscale-dev

# Install GStreamer development libraries
$ sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# Install additional codec and format libraries
$ sudo apt install libxvidcore-dev libx264-dev libmp3lame-dev libopus-dev

# Install additional audio codec libraries
$ sudo apt install libmp3lame-dev libvorbis-dev

# Install FFmpeg (which includes libavresample functionality)
$ sudo apt install ffmpeg

# Optional: Install VA-API for hardware acceleration
$ sudo apt install libva-dev

$ sudo apt install -y libgtk2.0-dev libgtkglext1-dev
```

- OpenCVは、GTK2とgtkglextを使用してOpenGLサポートしてる。
  しかし、システムにGTK3がインストールされている場合、gtkglextが検出されず、
  OpenGLサポートが無効になることがある。 

## Clone Opencv

download opencv with git repository

```
$ git clone https://github.com/opencv/opencv.git
$ git clone https://github.com/opencv/opencv_contrib.git
$ git checkout 4.10.0
$ git checkout 4.10.0
```

## Build and Install

```
$ cd opencv
$ mkdir build && cd build
```

```
cmake \
-D CMAKE_C_COMPILER=/usr/bin/gcc-10 \
-D CMAKE_CXX_COMPILER=/usr/bin/g++-10 \
-D CMAKE_BUILD_TYPE=Release \
-D WITH_CUDA=ON \
-D BUILD_opencv_cudacodec=OFF \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D WITH_GTK=ON \
-D CUDA_ARCH_BIN=8.6 \
-D OPENCV_EXTRA_MODULES_PATH=/home/kota/Projects/opencv_build/opencv_contrib/modules \
-D PYTHON3_EXECUTABLE=/home/kota/Work/imshow/.venv/bin/python \
-D OPENCV_PYTHON3_INSTALL_PATH=/home/kota/Work/imshow/.venv/lib/python3.10/site-packages/ \
-D BUILD_opencv_python3=ON \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D OPENCV_GENERATE_PKGCONFIG=ON ..
```

```
$ make -j$(nproc)
$ sudo make install
$ opencv_version
4.10.0
```

```
-- General configuration for OpenCV 4.10.0 =====================================
--   Version control:               4.10.0
-- 
--   Extra modules:
--     Location (extra):            /home/kota/Projects/opencv_build/opencv_contrib/modules
--     Version control (extra):     4.10.0
-- 
--   Platform:
--     Timestamp:                   2025-01-12T08:12:12Z
--     Host:                        Linux 6.8.0-51-generic x86_64
--     CMake:                       3.22.1
--     CMake generator:             Unix Makefiles
--     CMake build tool:            /usr/bin/gmake
--     Configuration:               Release
-- 
--   CPU/HW features:
--     Baseline:                    SSE SSE2 SSE3
--       requested:                 SSE3
--     Dispatched code generation:  SSE4_1 SSE4_2 FP16 AVX AVX2 AVX512_SKX
--       requested:                 SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX
--       SSE4_1 (18 files):         + SSSE3 SSE4_1
--       SSE4_2 (2 files):          + SSSE3 SSE4_1 POPCNT SSE4_2
--       FP16 (1 files):            + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 AVX
--       AVX (9 files):             + SSSE3 SSE4_1 POPCNT SSE4_2 AVX
--       AVX2 (38 files):           + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 FMA3 AVX AVX2
--       AVX512_SKX (8 files):      + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 FMA3 AVX AVX2 AVX_512F AVX512_COMMON AVX512_SKX
-- 
--   C/C++:
--     Built as dynamic libs?:      YES
--     C++ standard:                11
--     C++ Compiler:                /usr/bin/g++-10  (ver 10.5.0)
--     C++ flags (Release):         -fsigned-char -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -O3 -DNDEBUG  -DNDEBUG
--     C++ flags (Debug):           -fsigned-char -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -g  -O0 -DDEBUG -D_DEBUG
--     C Compiler:                  /usr/bin/gcc-10
--     C flags (Release):           -fsigned-char -W -Wall -Wreturn-type -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -O3 -DNDEBUG  -DNDEBUG
--     C flags (Debug):             -fsigned-char -W -Wall -Wreturn-type -Waddress -Wsequence-point -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -g  -O0 -DDEBUG -D_DEBUG
--     Linker flags (Release):      -Wl,--exclude-libs,libippicv.a -Wl,--exclude-libs,libippiw.a   -Wl,--gc-sections -Wl,--as-needed -Wl,--no-undefined  
--     Linker flags (Debug):        -Wl,--exclude-libs,libippicv.a -Wl,--exclude-libs,libippiw.a   -Wl,--gc-sections -Wl,--as-needed -Wl,--no-undefined  
--     ccache:                      NO
--     Precompiled headers:         NO
--     Extra dependencies:          m pthread cudart_static dl rt nppc nppial nppicc nppidei nppif nppig nppim nppist nppisu nppitc npps cublas cudnn cufft -L/usr/lib/x86_64-linux-gnu
--     3rdparty dependencies:
-- 
--   OpenCV modules:
--     To be built:                 alphamat aruco bgsegm bioinspired calib3d ccalib core cudaarithm cudabgsegm cudafeatures2d cudafilters cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping cudev datasets dnn dnn_objdetect dnn_superres dpm face features2d flann freetype fuzzy gapi hdf hfs highgui img_hash imgcodecs imgproc intensity_transform line_descriptor mcc ml objdetect optflow phase_unwrapping photo plot python3 quality rapid reg rgbd saliency sfm shape signal stereo stitching structured_light superres surface_matching text tracking ts video videoio videostab wechat_qrcode xfeatures2d ximgproc xobjdetect xphoto
--     Disabled:                    cudacodec world
--     Disabled by dependency:      -
--     Unavailable:                 cannops cvv java julia matlab ovis python2 viz
--     Applications:                tests perf_tests apps
--     Documentation:               NO
--     Non-free algorithms:         NO
-- 
--   GUI:                           GTK3
--     GTK+:                        YES (ver 3.24.33)
--       GThread :                  YES (ver 2.72.4)
--       GtkGlExt:                  NO
--     VTK support:                 NO
-- 
--   Media I/O: 
--     ZLib:                        /usr/lib/x86_64-linux-gnu/libz.so (ver 1.2.11)
--     JPEG:                        /usr/lib/x86_64-linux-gnu/libjpeg.so (ver 80)
--     WEBP:                        build (ver encoder: 0x020f)
--     PNG:                         /usr/lib/x86_64-linux-gnu/libpng.so (ver 1.6.37)
--     TIFF:                        /usr/lib/x86_64-linux-gnu/libtiff.so (ver 42 / 4.3.0)
--     JPEG 2000:                   build (ver 2.5.0)
--     OpenEXR:                     /usr/lib/x86_64-linux-gnu/libImath-2_5.so /usr/lib/x86_64-linux-gnu/libIlmImf-2_5.so /usr/lib/x86_64-linux-gnu/libIex-2_5.so /usr/lib/x86_64-linux-gnu/libHalf-2_5.so /usr/lib/x86_64-linux-gnu/libIlmThread-2_5.so (ver 2_5)
--     HDR:                         YES
--     SUNRASTER:                   YES
--     PXM:                         YES
--     PFM:                         YES
-- 
--   Video I/O:
--     DC1394:                      YES (2.2.6)
--     FFMPEG:                      YES
--       avcodec:                   YES (58.134.100)
--       avformat:                  YES (58.76.100)
--       avutil:                    YES (56.70.100)
--       swscale:                   YES (5.9.100)
--       avresample:                NO
--     GStreamer:                   YES (1.20.3)
--     v4l/v4l2:                    YES (linux/videodev2.h)
-- 
--   Parallel framework:            pthreads
-- 
--   Trace:                         YES (with Intel ITT)
-- 
--   Other third-party libraries:
--     Intel IPP:                   2021.11.0 [2021.11.0]
--            at:                   /home/kota/Projects/opencv_build/opencv/build/3rdparty/ippicv/ippicv_lnx/icv
--     Intel IPP IW:                sources (2021.11.0)
--               at:                /home/kota/Projects/opencv_build/opencv/build/3rdparty/ippicv/ippicv_lnx/iw
--     VA:                          YES
--     Lapack:                      NO
--     Eigen:                       YES (ver 3.4.0)
--     Custom HAL:                  NO
--     Protobuf:                    build (3.19.1)
--     Flatbuffers:                 builtin/3rdparty (23.5.9)
-- 
--   NVIDIA CUDA:                   YES (ver 11.5, CUFFT CUBLAS)
--     NVIDIA GPU arch:             86
--     NVIDIA PTX archs:
-- 
--   cuDNN:                         YES (ver 8.2.4)
-- 
--   OpenCL:                        YES (INTELVA)
--     Include path:                /home/kota/Projects/opencv_build/opencv/3rdparty/include/opencl/1.2
--     Link libraries:              Dynamic load
-- 
--   Python 3:
--     Interpreter:                 /home/kota/Work/imshow/.venv/bin/python (ver 3.10.12)
--     Libraries:                   /usr/lib/x86_64-linux-gnu/libpython3.10.so (ver 3.10.12)
--     Limited API:                 NO
--     numpy:                       /home/kota/Work/imshow/.venv/lib/python3.10/site-packages/numpy/_core/include (ver 2.2.1)
--     install path:                /home/kota/Work/imshow/.venv/lib/python3.10/site-packages//cv2/python-3.10
-- 
--   Python (for build):            /home/kota/Work/imshow/.venv/bin/python
-- 
--   Java:                          
--     ant:                         NO
--     Java:                        NO
--     JNI:                         NO
--     Java wrappers:               NO
--     Java tests:                  NO
-- 
--   Install to:                    /usr/local
-- -----------------------------------------------------------------
```


```python
import cv2
import numpy as np

def main():
    # 1. 適当なサイズのランダム画像を作成 (CPU上のnumpy配列)
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # 2. GPU メモリへアップロード (cv2.cuda_GpuMat オブジェクトを生成してデータを upload)
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)

    # 3. GPU 上でガウシアンフィルタを作成・適用
    #    createGaussianFilter(入力タイプ, 出力タイプ, カーネルサイズ, sigmaX, sigmaY=0, 境界オプション...)
    gaussian_filter = cv2.cuda.createGaussianFilter(
        srcType=cv2.CV_8UC3,
        dstType=cv2.CV_8UC3,
        ksize=(5, 5),
        sigma1=5
    )
    gpu_blurred = gaussian_filter.apply(gpu_img)

    # 4. GPU メモリから CPU へダウンロード (numpy配列として取り出す)
    blurred_img = gpu_blurred.download()

    # 結果をウィンドウで表示
    cv2.imshow("Original (CPU)", img)
    cv2.imshow("Blurred (GPU)", blurred_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

## 参考

https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7
https://gist.github.com/minhhieutruong0705/8f0ec70c400420e0007c15c98510f133
https://gist.github.com/madtunebk/5f20437725eb0e0cfc2a4934153b0ab4
https://github.com/alexfcoding/OpenCV-cuDNN-manual