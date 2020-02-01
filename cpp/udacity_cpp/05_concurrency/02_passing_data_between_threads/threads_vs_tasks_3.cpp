#include <iostream>
#include <thread>
#include <future>
#include <cmath>
#include <vector>
#include <chrono>

int workerFunction(int n)
{
    std::cout << "Woker thread id = " << std::this_thread::get_id()
              << std::endl;

    double result = 0;
    for (int i = 0; i < n; ++i) {
        result += sqrt(12345.6789);
    }
    return (int)result;
}

int main()
{
    std::cout << "Main thread id = " << std::this_thread::get_id()
              << std::endl;

    std::chrono::high_resolution_clock::time_point t1 = 
                                    std::chrono::high_resolution_clock::now();
    std::vector<std::future<int>> futures;
    int n_loops = 10, n_threads = 5;
    for (int i = 0; i < n_threads; i++) {
        futures.emplace_back(std::async(std::launch::any, workerFunction, n_loops));
    }

    for (const std::future<int> &ftr: futures)
        ftr.wait();
    for (std::future<int> &ftr: futures)
        std::cout << "result: " << ftr.get() << std::endl;

    std::chrono::high_resolution_clock::time_point t2 = 
                                    std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "Execution finished after " << duration << "microseconds" << std::endl;
    
    return 0;
}
