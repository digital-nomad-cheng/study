#include <iostream>
#include <thread>
#include <vector>
#include <future>
#include <mutex>

std::mutex mtx;
double result;

void printResult(int denom)
{
    std::cout << "for denom = " << denom << ", the result is " << result 
              << std::endl;
}

void divideByNumber(double num, double denom)
{
    mtx.lock();
    try {
        if (denom != 0) {
            result = num / denom;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            printResult(denom);
        } else {
            throw std::invalid_argument("Exception from thread: Division by zero!");
        }
    } catch (const std::invalid_argument &e) {
        std::cout << e.what() << std::endl;
        return;
    }
    mtx.unlock();
}

int main()
{
    std::vector<std::future<void>> futures;
    for (double i = 1; i <= 10; i++) {
        futures.emplace_back(std::async(std::launch::async, divideByNumber, 50.0, i));
    }

    std::for_each(futures.begin(), futures.end(), [](std::future<void> &ftr) {
        ftr.wait();
    });

    return 0;
}

