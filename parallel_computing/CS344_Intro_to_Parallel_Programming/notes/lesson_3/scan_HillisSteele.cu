#include <cstdio>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void scanHillisSteele(int *d_out, int *d_in, int n)
{
  int idx = threadIdx.x;
  extern __shared__ int tmp[];
  int pout = 0, pin = 1;
  
  tmp[idx] = (idx > 0) ? d_in[idx-1] : 0;
  __syncthreads();

  for (int offset = 1; offset < n; offset *=2)
  {
    // swap double buffer indices
    pout = 1 - pout;
    pin = 1 - pout;

    if (idx >= offset) {
      tmp[pout*n + idx] = tmp[pin*n + idx - offset] + tmp[pin*n + idx];
    } else {
      tmp[pout*n + idx] = tmp[pin*n + idx];
    }

    __syncthreads();
  }
    d_out[idx] = tmp[pout*n + idx];
}

int main()
{
  const int ARRAY_SIZE = 10;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
  int h_in[ARRAY_SIZE]{1, 2, 5, 7, 8, 10, 11, 12, 15, 19};
  int h_out[ARRAY_SIZE];

  int *d_in;
  int *d_out;

  cudaMalloc((void **)&d_in, ARRAY_BYTES);
  cudaMalloc((void **)&d_out, ARRAY_BYTES);

  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

  scanHillisSteele<<<1, ARRAY_SIZE, 2*ARRAY_BYTES>>>(d_out, d_in, ARRAY_SIZE);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  std::cout << "Input: " << std::endl;
  for (int i = 0; i < ARRAY_SIZE; i++) {
    std::cout << h_in[i] << " " << std::endl;
  }
  std::cout << "Exclusive scan with operation +;" << std::endl;
  for (int i = 0; i < ARRAY_SIZE; i++) {
    std::cout << h_out[i] << " " << std::endl;
  }

  cudaFree(d_in);
  cudaFree(d_out);

 }
