#include <iostream>
#include <thread>
#include <vector>
#include <future>
#include <mutex>

class Vehicle
{
public:
    Vehicle(int id): _id(id) {}
    int getID() { return _id; }
private:
    int _id;
};

class WaitingVehicles
{
public:
    WaitingVehicles() {}
    void printSize()
    {
        _mutex.lock();
        std::cout << "#vehicles = " << _vehicles.size() << std::endl;
        _mutex.unlock();
    }

    void pushBack(Vehicle &&v)
    {
        for (size_t i = 0; i < 3; i++) {
            if (_mutex.try_lock_for(std::chrono::milliseconds(100))) {
                _vehicles.emplace_back(std::move(v));
                _mutex.unlock();
                break;
            } else {
                std::cout << "Error! Vehicle #" << v.getID() 
                          << " could not be added to vector" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

private:
    std::vector<Vehicle> _vehicles;
    std::timed_mutex _mutex;
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





