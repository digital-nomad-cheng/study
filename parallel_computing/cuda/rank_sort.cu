#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

#define arraySize 5
#define threadsPerBlock 5

__global__ void addKernel(int *d_a, int *d_b)
{
    int count = 0;
    int tid = threadIdx.x;
    int ttid = blockIdx.x*threadsPerBlock + tid;
    int val = d_a[ttid];
    __shared__ int cache[threadsPerBlock];
    for (int i = tid; i < arraySize; i += threadsPerBlock) {
        cache[tid] = d_a[i];
        __syncthreads();
        for (int j = 0; j < threadsPerBlock; j++) {
            if (val > cache[j])
                count++;
        }
        __syncthreads();
    }
    d_b[count] = val;
}

int main()
{
    int h_a[arraySize] = {5, 9, 3, 4, 8};
    int h_b[arraySize];
    int *d_a, *d_b;

    cudaMalloc((void**)&d_b, arraySize*sizeof(int));
    cudaMalloc((void**)&d_a, arraySize*sizeof(int));

    cudaMemcpy(d_a, h_a, arraySize*sizeof(int), cudaMemcpyHostToDevice);

    addKernel << < arraySize/threadsPerBlock, threadsPerBlock>> >(d_a, d_b);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_b, d_b, arraySize*sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("The Enumeration sorted Array is:\n");
    for (int i = 0; i < arraySize; i++)
        printf("%d\n", h_b[i]);
    
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}  

