cmake_minimum_required(VERSION 3.16)
project(HandCursor)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

# include has 2 subdirectories: tensorflow and flatbuffers
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../tflite-dist/include/ ${OpenCV_INCLUDE_DIRS})

# lib has 1 file: libtensorflowlite.so
ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../tflite-dist/libs/linux_x64/libtensorflowlite.so)

add_executable(HandCursor.o main.cpp)
target_link_libraries(HandCursor.o tensorflowlite ${OpenCV_LIBS})
