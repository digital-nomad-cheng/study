#include <memory>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring> // memset

int main(void)
{
    int device;
    cudaDeviceProp device_property;
    cudaGetDevice(&device);
    printf("ID of device: %d\n", device);
    memset(&device_property, 0, sizeof(cudaDeviceProp));
    device_property.major = 1;
    device_property.minor = 3;
    cudaChooseDevice(&device, &device_property);
    printf("ID of device which supports double precision is: %d\n", device);
    cudaSetDevice(device);
}
