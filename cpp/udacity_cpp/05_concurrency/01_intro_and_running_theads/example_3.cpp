#include <iostream>
#include <thread>

void threadFunction()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "Finished work in thread." << std::endl;
}

int main()
{
    std::thread t(threadFunction);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::cout << "Finished work in main." << std::endl;

    t.join();

    return 0;
}

