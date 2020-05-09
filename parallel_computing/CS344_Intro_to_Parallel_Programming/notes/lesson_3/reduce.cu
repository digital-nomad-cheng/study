#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

__global__ void globalMem_reduce_kernel(float *d_out, float *d_in)
{
    int ttid = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { // if threadIdx.x is on the left half
            d_in[ttid] += d_in[ttid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        d_out[blockIdx.x] = d_in[ttid];
    }
}


__global__ void sharedMem_reduce_kernel(float *d_out, float *d_in)
{
    // shared data is allocated in the kernel call: 3rd argument
    extern __shared__ float shared_data[];
    int ttid = threadIdx.x + blockDim.x * threadIdx.x;
    int tid = threadIdx.x;

    // load shared memory from global memory
    shared_data[tid] = d_in[ttid];
    __syncthreads();

    // reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid+s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = shared_data[0];
    }
}

void reduce(float *d_out, float *d_intermediate, float *d_in, int size, 
        bool useSharedMem)
{
    // assumption 1: size is not greater than maxThreadsPerBlock**2
    // assumption 2: size is a multiple of maxThreadsPerBlock

    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;
    // int blocks = size % maxThreadsPerBlock ? 
    //    (size / maxThreadsPerBlock + 1) : (size / maxThreadsPerBlock);
    int blocks = size / maxThreadsPerBlock;

    if (useSharedMem) {
        sharedMem_reduce_kernel<<<blocks, threads, threads*sizeof(float)>>>(d_intermediate, d_in);
    } else {
        globalMem_reduce_kernel<<<blocks, threads>>>(d_intermediate, d_in);
    }

    threads = blocks;
    blocks = 1;

    if (useSharedMem) {
        sharedMem_reduce_kernel<<<blocks, threads, threads*sizeof(float)>>>(d_out, d_intermediate);
    } else {
        globalMem_reduce_kernel<<<blocks, threads>>>(d_out, d_intermediate);
    }
}


int main(int argc, char **argv)
{
    // --- checking whether there is a device --- //
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No GPUs found" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // --- get properties of device --- //
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProps;
    if (cudaGetDeviceProperties(&deviceProps, dev) == 0) {
        std::cout << "Using device:" << dev << "\n";
        std::cout << deviceProps.name << "\n";
        std::cout << "Global memory: " << deviceProps.totalGlobalMem << "\n";
        std::cout << "Compute v:" << static_cast<int>(deviceProps.major) << "."
            << static_cast<int>(deviceProps.minor) << std::endl;
        std::cout << "Clock:" << static_cast<int>(deviceProps.clockRate) << std::endl;
    }

    const int ARRAY_SIZE = 1 << 20;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    float h_in[ARRAY_SIZE];
    float sum = 0.0f;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = -1.0f + static_cast<float>(random()) / (static_cast<float>(RAND_MAX)/2.0f);
        sum += h_in[i];
    }

    float *d_in, *d_intermediate, *d_out;
    cudaMalloc((void **)&d_in, ARRAY_BYTES);
    cudaMalloc((void **)&d_intermediate, ARRAY_BYTES);
    cudaMalloc((void **)&d_out, sizeof(float));

    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    int whichKernel = 0;
    if (argc == 2) {
        whichKernel = atoi(argv[1]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    switch (whichKernel) {
        case 0:
            std::cout << "Global memory reduce" << "\n";
            cudaEventRecord(start, 0);
            for (int i = 0; i < 100; i++) {
                reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, false);
            }
            cudaEventRecord(stop, 0);
            break;
        case 1:
            std::cout << "Shared memory reduce" << "\n";
            cudaEventRecord(start, 0);
            for (int i = 0; i < 100; i++) {
                reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, true);
            }
            cudaEventRecord(stop, 0);
            break;
        default:
            std::cerr << "No kernel run!" << std::endl;
            exit(EXIT_FAILURE);
    }

    // calculate elapsedTime
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    elapsed /= 100.0f;

    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Everage time elapsed:" << elapsed << std::endl;
    std::cout << "Host result:" << sum << ", device result:" << h_out << std::endl;

    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);
}
