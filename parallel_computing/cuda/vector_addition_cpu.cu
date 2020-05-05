#include <cstdio>
#include <iostream>
#include <chrono>
#define N 5000

void cpuAdd(int *h_a, int *h_b, int *h_c)
{
    int tid = 0;
    while (tid < N)
    {
        h_c[tid] = h_a[tid] + h_b[tid];
        tid += 1;
    }
}

int main(void)
{
    int h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++)
    {
        h_a[i] = 2 * i * i;
        h_b[i] = i;
    }
    auto t0 = std::chrono::steady_clock::now();
    cpuAdd(h_a, h_b, h_c);
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "CPU time:" << std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() << "\n";
    printf("Vector addition on CPU\n");
    // for (int i = 0; i < N; i++)
    // {
    //    printf("The sum of %d element is %d + %d = %d\n", i, h_a[i], h_b[i], h_c[i]);
    // }
    return 0;
}
