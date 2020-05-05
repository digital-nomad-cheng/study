#include <cstdio>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 500

__global__ void gpuAdd(int *d_a, int *d_b, int *d_c)
{
    // Getting block index of current kernel
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    while (tid < N)
    {
        d_c[tid] = d_a[tid] + d_b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main()
{
    int h_a[N], h_b[N], h_c[N];
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N*sizeof(int));
    cudaMalloc((void**)&d_b, N*sizeof(int));
    cudaMalloc((void**)&d_c, N*sizeof(int));
    for (int i = 0; i < N; i++) {
        h_a[i] = 2 * i * i;
        h_b[i] = i;
    }
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice);
    
    gpuAdd << <512, 512>> > (d_a, d_b, d_c);
    
    cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    int correct = 1;
    for (int i = 0; i < N; i++) {
        if ((h_a[i] + h_b[i]) != h_c[i]) {
            // printf("h_a[%d] = %d, h_b[%d] = %d, h_c[%d] = %d\n", 
            //        i, h_a[i], i, h_b[i], i, h_c[i]);
            correct = 0;
        }
    }

    if (correct == 1) 
        printf("GPU has computed sum correctedly\n");
    else
        printf("GPU has failed to compute sum\n");
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}
