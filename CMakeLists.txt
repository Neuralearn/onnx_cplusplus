
cmake_minimum_required(VERSION 3.13)
project(onnx_c++)


find_package(OpenCV REQUIRED)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

add_executable(onnx_c++ main.cpp)
target_include_directories(onnx_c++ PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(onnx_c++ PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})