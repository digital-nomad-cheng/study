cmake_minimum_required(VERSION 3.10)
set(CMAKE_C_COMPILER "gcc-6")
project(image_blur)

find_package(CUDA REQUIRED)
set(OpenCV_DIR "/home/vincent/Downloads/LibModels/opencv-4.3.0/build")
set(OpenCV_INCLUDE_DIR "/home/vincent/Libs/opencv/include/opencv4")

find_package(OpenCV REQUIRED)

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW2_files main.cpp reference_calc.cpp compare.cpp)
    
CUDA_ADD_EXECUTABLE(HW2 ${HW2_files} ${hdr} ${cu})

target_link_libraries(HW2 ${OpenCV_LIBS})

