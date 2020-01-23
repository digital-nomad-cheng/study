/* Templates
With templates, the idea is to pass a data type as a parameter so that you don’t
need to write the same function code for operating on different data types.

For example, you might need a function that has to accept many different data
types in order for it to perform some operations or other actions on those data
types.

Some of these actions can be dividing, sorting, etc.
Rather than writing and maintaining the multiple function declarations, each 
accepting slightly different arguments, you can write one function and pass the
argument types as parameters.

At compile time, the compiler then expands the code using the types that are 
passed as parameters.


So if you write:
    template <typename Type>
        Type Sum(Type a, Type b) {
            return a+b;
        }
    int main() {
        std::cout << Sum<double>(20.0, 13.7) << std::endl;

The compiler adds the following internally:
    double Sum(double a, double b) {
        return a+b;
    }
Or in this case:

std::cout << Sum<char> (‘Z’, ’j’) << std::endl;
The compiler adds:
    char Sum(char a, char b) {
        return a+b;
    }
In stead of typename, C++ also
supports the keyword class ( template <class T>), but using typename is preferred.
Like normal parameters, you can specify the default arguments to templates.

*/

#include <iostream>
#include <typeinfo>

template<typename T, typename U = int>

class A
{
public:
    T x;
    U y;
    A()
    {
        std::cout << "Constructor called" << "\n";
        std::cout << "x type: " << typeid(x).name() << std::endl;
        std::cout << "y type: " << typeid(y).name() << std::endl;
    }
};

template <typename type>
type sum(type a, type b)
{
    return a + b;
}
template <typename T1, typename T2>
T1 sum(T1 a, T2 b)
{
    return a + b;
}

int main()
{
    A<char, char> obj1;
    A<char> obj2;
    
    std::cout << sum<double>(20.0, 13.7) << std::endl;
    std::cout << sum<double, int>(20.1, 13) << std::endl;
}

    
