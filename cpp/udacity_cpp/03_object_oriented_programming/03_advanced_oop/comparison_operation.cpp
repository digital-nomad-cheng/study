#include <iostream>
#include <typeinfo>

template <typename T>
T returnMax(T a, T b)
{
    if (a > b)
        return a;
    else
        return b;
}

int main()
{
    int num1 = 50;
    int num2 = 10;

    int res = returnMax(num1, num2);
    std::cout << "Bigger: " << res << std::endl;
    
    double n1 = 45.65;
    double n2 = 100.45;

    double res2 = returnMax(n1, n2);
    std::cout << "Bigger one is: " << res2 << std::endl;
}

