#include <iostream>
#include <string>
#include <assert.h>

using namespace std;

class Animal
{
public:
    double age;
};

class Pet
{
public:
    string name;
};

class Dog: public Animal, public Pet
{
public:
    string breed;
};

class Cat: public Animal, public Pet
{
public:
    Cat(string name, double age, string color)
    {
        this->color = color;
        this->name = name;
        this->age = age;
    }
    string color;

    void getInfo()
    {
        std::cout << name << std::endl;
        std::cout << age << std::endl;
        std::cout << color << std::endl;
    }
};

int main()
{
    Cat cat("Max", 10, "black");
    assert(cat.color == "black");
    assert(cat.age == 10);
    assert(cat.name == "Max");
    cat.getInfo();
}

