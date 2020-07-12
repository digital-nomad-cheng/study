cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER "gcc-6")

project(image_HDR)

find_package(CUDA REQUIRED)
set(OpenCV_DIR "/home/vincent/Downloads/LibModels/opencv-4.3.0/build")
set(OpenCV_INCLUDE_DIR "/home/vincent/Libs/opencv/include/opencv4")
find_package(OpenCV REQUIRED)

SET (compare_files compare.cpp)

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW3_files main.cpp loadSaveImage.cpp reference_calc.cpp compare.cpp)
    
CUDA_ADD_EXECUTABLE(HW3 ${HW3_files} ${hdr} ${cu})

target_link_libraries(HW3 ${OpenCV_LIBS})
