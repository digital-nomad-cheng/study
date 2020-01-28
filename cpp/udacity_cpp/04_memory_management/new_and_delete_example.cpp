#include <iostream>

int main()
{
    int *ptr = nullptr;
    
    ptr = new(std::nothrow) int;
    if (!ptr) {
        std::cout << "Mem alloc failed!" << std::endl;
    } else {
        *ptr = 31;
        std::cout << " Address is: " << ptr << std::endl;
        std::cout << " Value is: " << *ptr << std::endl;
    }

    int *arr_ptr = new(std::nothrow) int[3];

    if (!arr_ptr) 
        throw("Failed to allocate memory for array");

    for (int i = 0; i < 3; i++) {
        arr_ptr[i] = i*i;
        std::cout << "Address is: " << &arr_ptr[i] << std::endl;
        std::cout << "Value is: " << arr_ptr[i] << std::endl;
    }

    delete ptr;
    delete [] arr_ptr;

    return 0;
}

