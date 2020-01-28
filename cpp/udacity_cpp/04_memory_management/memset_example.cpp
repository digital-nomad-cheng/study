#include <cstring>
#include <iostream>

int main()
{
    char *ptr = new char('C');
    char str1[4] = "AAA";
    char str2[4] = "AAA";

    std::cout << "Before Value of *ptr: " << *ptr << std::endl;
    std::cout << "Before Value of str1: " << str1 << std::endl;
    std::cout << "Before Value of str2: " << str2 << std::endl;

    memset(ptr, 'A', 1);
    memset(str1+1, 'B', 1);
    memset(str2+1, 'B', 2);

    
    std::cout << "After Value of *ptr: " << *ptr << std::endl;
    std::cout << "After Value of str1: " << str1 << std::endl;
    std::cout << "After Value of str2: " << str2 << std::endl;

    delete ptr;

    return 0;
}


