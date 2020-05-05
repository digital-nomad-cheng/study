#include <memory>
#include <iostream>
#include <cuda_runtime.h>

int main(void)
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "There are " << device_count << " gpus on this computer" << std::endl;
}
