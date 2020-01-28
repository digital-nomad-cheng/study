#include <iostream>

int main()
{
    int *ptr;
    ptr = new int;
    *ptr = 25;
    std::cout << "Address of a pointer: " << ptr << std::endl;
    std::cout << "Value of a pointer: " << *ptr << std::endl;

    delete ptr;
    return 0;
}

