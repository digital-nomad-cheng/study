#include <iostream>
#include <thread>
#include <future>

class Vehicle
{
public:
    Vehicle(): _id(0)
    {
        std::cout << "Vehicle #" << _id << "Default constructor called" << std::endl;
    }
    Vehicle(int id): _id(id)
    {
        std::cout << "Vehicle #" << _id << "Initializing constructor called" << std::endl;
    }

    void setID(int id) { _id = id; }
    int getID() { return _id; }

private:
    int _id;
};

int main()
{
    Vehicle v0;
    Vehicle v1(1);
    for (int i = 0; i < 1000; i++) {
        std::future<void> ftr = std::async([](Vehicle v) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            v.setID(2);
        }, v0);

        v0.setID(3);

        ftr.wait();
        std::cout << "Vehicle #" << v0.getID() << std::endl;
    }
    return 0;
}
