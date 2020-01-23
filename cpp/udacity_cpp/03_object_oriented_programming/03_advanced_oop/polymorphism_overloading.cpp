/*
Polymorphism
Polymorphism is means "assuming many forms".
In the context of object-oriented programming, polymorphism) describes a 
paradigm in which a function may behave differently depending on how it is 
called. In particular, the function will perform differently based on its inputs.
Polymorphism can be achieved in two ways in C++: overloading and overriding. In 
this exercise we will focus on overloading.

Overloading
Overloading can happen inside class and in function.

*/
#include <iostream>
#include <string>
#include <ctime>

using namespace std;

class Date
{
public:
    Date(int day, int month, int year): day(day), month(month), year(year) {};
    Date(int day, int month): day(day), month(month)
    {
        time_t t = time(NULL);
        tm *time_ptr = localtime(&t);
        year = time_ptr->tm_year;
    }
    void printDate()
    {
        std::cout << year << "/" << month << "/" << day << std::endl;
    }
private:
    int day;
    int month;
    int year;
};


void hello()
{
    std::cout << "Hello, World" << std::endl;
}

void hello(string name)
{
    std::cout << "Hello: " << name << std::endl;
}

int main()
{
    Date dt1(1, 1, 1);
    Date dt2(1, 1);
    dt1.printDate();
    dt2.printDate();
    hello();
    hello("cat");
}

