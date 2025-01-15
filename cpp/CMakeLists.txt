cmake_minimum_required(VERSION 3.10)
project(imshow)

# --------------------------------------------------
# 1) Apt でインストールした OpenCV を使う場合
# --------------------------------------------------
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
#
# --------------------------------------------------
# 2) ソースコードからビルドした OpenCV を使う場合
# --------------------------------------------------
# set(OpenCV_DIR "/usr/local/include/opencv4")

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

# OpenCVライブラリの情報を表示（デバッグ用）
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# 実行ファイルを生成
add_executable(imshow imshow.cpp)

target_link_libraries(imshow PRIVATE ${OpenCV_LIBS} fmt::fmt)

