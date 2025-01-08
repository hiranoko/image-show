cmake_minimum_required(VERSION 3.10)
project(imshow)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

# 実行ファイルを生成
add_executable(imshow imshow.cpp)

target_link_libraries(imshow PRIVATE ${OpenCV_LIBS} fmt::fmt)

