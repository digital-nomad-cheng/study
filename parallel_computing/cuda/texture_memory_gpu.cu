#include <cstdio>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_THREADS 10
#define N 10

// first argument: data type of texture elements
// second argument: types of texture reference which can be one-dimensional, two-dimensional...
// third argument: read mode, optional 
texture <float, 1, cudaReadModeElementType> textureRef;
__global__ void gpu_texture_memory(int n, float *d_out)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N) {
        float temp = tex1D(textureRef, float(idx));
        d_out[idx] = temp;
    }
}

int main()
{
    int num_blocks = N / NUM_THREADS + ((N % NUM_THREADS) ? 1 : 0);
    float *d_out;

    cudaMalloc((void**)&d_out, sizeof(float)*N);
    float h_out[N], h_in[N];
    for (int i = 0; i < N; i++) {
        h_in[i] = float(i);
    }
    
    // Define cuda array which is dedicated to textures compared to normal array
    cudaArray *cu_array;
    cudaMallocArray(&cu_array, &textureRef.channelDesc, N, 1);
    // copy data to cuda array
    // 0, 0 meaning starting from the top left corner
    cudaMemcpyToArray(cu_array, 0, 0, h_in, N*sizeof(float), cudaMemcpyHostToDevice);

    // bind a texture to the CUDA array
    cudaBindTextureToArray(textureRef, cu_array);
    
    gpu_texture_memory << <num_blocks, NUM_THREADS>> > (N, d_out);

    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Use of texture memory on GPU: \n");
    for (int i = 0; i < N; i++) {
        printf("Texture element at %d is: %f\n", i, h_out[i]);
    }

    cudaFree(d_out);
    cudaFreeArray(cu_array);
    cudaUnbindTexture(textureRef);

}
