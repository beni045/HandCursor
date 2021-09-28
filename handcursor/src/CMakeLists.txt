cmake_minimum_required(VERSION 3.15)
project(HandCursor)

set(CMAKE_CXX_STANDARD 14)

# include has 2 subdirectories: tensorflow and flatbuffers
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../tflite-dist/include/ ${OpenCV_INCLUDE_DIRS} ../inc)

find_package(OpenCV REQUIRED)

# lib has 3 file: tensorflowlite.dll.if.lib flatbuffers.lib tensorflowlite.dll
find_library(TFLITE tensorflowlite.dll.if HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../tflite-dist/libs/windows_x64/ )
find_library(FLATBUFFERS flatbuffers  HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../tflite-dist/libs/windows_x64/ )

add_executable(HandCursor main.cpp modelprocessor.cpp handdetector.cpp keypointdetector.cpp utils.cpp)
target_link_libraries(HandCursor PRIVATE ${TFLITE} ${FLATBUFFERS} ${OpenCV_LIBS})

add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../../tflite-dist/libs/windows_x64/tensorflowlite.dll ${CMAKE_BINARY_DIR}/Release/
)