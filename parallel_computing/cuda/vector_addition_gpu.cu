#include <cstdio>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 5000
__global__ void gpuAdd(int *d_a, int *d_b, int *d_c)
{
    int tid = blockIdx.x;
    if (tid < N)
        d_c[tid] = d_a[tid] + d_b[tid];
}

int main()
{
    int h_a[N], h_b[N], h_c[N];
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N*sizeof(int));
    cudaMalloc((void**)&d_b, N*sizeof(int));
    cudaMalloc((void**)&d_c, N*sizeof(int));
    for (int i = 0; i < N; i++) {
        h_a[i] = 2 * i *i;
        h_b[i] = i;
    }

    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice);
    using namespace std::chrono;
    auto t0 = steady_clock::now();
    gpuAdd << <N, 1>> >(d_a, d_b, d_c);
    auto t1 = steady_clock::now();
    std::cout << "GPU time:" << duration_cast<nanoseconds>(t1-t0).count() << "\n";
    cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    printf("Vector addition on GPU \n");
    // for (int i = 0; i < N; i++)
    // {
    //    printf("The sum of %d element is %d + %d = %d\n", i, h_a[i], h_b[i], h_c[i]);
    // }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;

}

    
