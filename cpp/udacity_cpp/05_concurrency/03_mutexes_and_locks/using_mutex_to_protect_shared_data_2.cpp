#include <iostream>
#include <thread>
#include <vector>
#include <future>
#include <mutex>

class Vehicle
{
public:
    Vehicle(int id): _id(id) {}
private:
    int _id;
};

class WaitingVehicles
{
public:
    WaitingVehicles(): _tmp_vehicles(0) {}
    void printSize()
    {
        _mutex.lock();
        std::cout << "#vehicles = " << _tmp_vehicles << std::endl;
        _mutex.unlock();
    }

    void pushBack(Vehicle &&v)
    {
        _mutex.lock();
        int old_num = _tmp_vehicles;
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        _tmp_vehicles = old_num + 1;
        _mutex.unlock();
    }

private:
    std::vector<Vehicle> _vehicles;
    int _tmp_vehicles;
    std::mutex _mutex;
};

int main()
{
    std::shared_ptr<WaitingVehicles> queue(new WaitingVehicles);
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 1000; ++i)
    {
        Vehicle v(i);
        futures.emplace_back(std::async(std::launch::async, 
            &WaitingVehicles::pushBack, queue, std::move(v)));
    }

    std::for_each(futures.begin(), futures.end(),
        [](std::future<void> &ftr) {
            ftr.wait();
    });

    queue->printSize();

    return 0;
}





