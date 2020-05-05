#include <cstdio>

__global__ void gpu_shared_memory(float *d_a)
{
    int i, index = threadIdx.x;
    float average, sum = 0.0f;
    
    // Define shared memory
    __shared__ float sh_arr[10];
    sh_arr[index] = d_a[index];
    // Directive ensure all the writes to shared memory have completed
    __syncthreads();
    for (int i = 0; i <= index; i++)
    {
        sum += sh_arr[i];
    }
    average = sum / (index + 1.0f);
    d_a[index] = average;
    // This statement is redundant and will have no effect on overall code execution
    // Why ???
    sh_arr[index] = average;
}

int main()
{
    float h_a[10];
    float *d_a;

    for (int i = 0; i < 10; i++) {
        h_a[i] = i;
    }

    cudaMalloc((void **)&d_a, sizeof(float)*10);
    cudaMemcpy(d_a, h_a, 10*sizeof(float), cudaMemcpyHostToDevice);
    gpu_shared_memory << <1, 10>> >(d_a);
    cudaMemcpy((void *)h_a, (void *)d_a, 10*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Use of shared Memory on GPU:\n");
    
    for (int i = 0; i < 10; i++) {
        printf("The running average after %d element is %f\n", i, h_a[i]);
    }
}

