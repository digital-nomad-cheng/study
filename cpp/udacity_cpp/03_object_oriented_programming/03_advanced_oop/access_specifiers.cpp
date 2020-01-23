#include <iostream>
#include <string>

using namespace std;


// privacy levels between parent and child classes
class Vehicle
{
public:
    int wheels = 0;
    string color = "blue";
    void print() const
    {
        cout << "This " << color << " vehicle has " << wheels << " wheels!\n";
    }
};

class Car : public Vehicle
{
public:
    bool sun_roof = false;
};

class Bicycle : protected Vehicle
{
public:
    bool kick_stand = true;
    void setWheels(int w)
    {
        wheels = w;
    }
};

class Scooter : private Vehicle
{
public:
    bool electric = false;
    void setWheels(int w)
    {
        wheels = w;
    }
};

class ScooterChild : public Bicycle // cannot be Scooter here
{
public:
    int getWheels() 
    {
        return wheels;
    }
};

int main()
{
    Car car;
    car.wheels = 4;
    Bicycle bicycle;
    // bicycle.wheels = 2;
    bicycle.setWheels(2);
    Scooter scooter;
    scooter.setWheels(2);
    
    ScooterChild sc;
    sc.getWheels();  
}
