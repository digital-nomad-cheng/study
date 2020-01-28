#include <iostream>
#include <memory>

class MyClass
{
public:
    void classMethod() {
        std::cout << "MyClass::classMethod()" << std::endl;
    }
};

int main()
{
    std::unique_ptr<MyClass> ptr_1 (new MyClass);
    ptr_1->classMethod();

    std::cout << ptr_1.get() << std::endl;
    // transfer ownership to ptr2
    std::unique_ptr<MyClass> ptr_2 = std::move(ptr_1);
    ptr_2->classMethod();

    std::cout << ptr_1.get() << std::endl;
    std::cout << ptr_2.get() << std::endl;

    return 0;
}

