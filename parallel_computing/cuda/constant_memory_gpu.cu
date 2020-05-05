#include <cstdio>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int constant_f;
__constant__ int constant_g;

#define N 5

__global__ void gpu_constant_memory(float *d_in, float *d_out)
{
    int tid = threadIdx.x;
    d_out[tid] = constant_f*d_in[tid] + constant_g;
}

int main()
{
    float h_in[N], h_out[N];
    float *d_in, *d_out;
    int h_f = 2;
    int h_g = 20;

    cudaMalloc((void**)&d_in, N*sizeof(float));
    cudaMalloc((void**)&d_out, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        h_in[i] = i;
    }

    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    // the last two parameters are optional
    cudaMemcpyToSymbol(constant_f, &h_f, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constant_g, &h_g, sizeof(int), 0, cudaMemcpyHostToDevice);

    gpu_constant_memory << <1, N>> >(d_in, d_out);
    
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++)
    {
        printf("The expression for input %f is %f\n", h_in[i], h_out[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);

}
