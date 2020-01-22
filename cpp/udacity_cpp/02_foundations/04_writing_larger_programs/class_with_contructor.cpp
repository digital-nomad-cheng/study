/*
Adding a Constructor
The best way to fix this is to add a constructor to the Car class.
The constructor allows you to instantiate new objects with the data that you want.
In the next code cell, we have added a constructor for Car that allows the number 
and color to be passed in. This means that each Car object can be created with 
those variables.
*/

#include <iostream>
#include <string>

using namespace std;

class Car
{
public:
    void printCarData()
    {
        cout << "The distance that the " << color << " car" << number << " has " 
        "travalled is: " << distance << endl;
    }

    void incrementDistance()
    {
        distance++;
    }

    // constructor
    Car(string c, int n)
    {
        color = c;
        number = n;
    }

    string color;
    int distance = 0;
    int number;
};

int main()
{
    Car car_1 = Car("green", 1);
    Car car_2 = Car("red", 2);
    Car car_3 = Car("blue", 3);

    car_1.incrementDistance();

    car_1.printCarData();
    car_2.printCarData();
    car_3.printCarData();

}
