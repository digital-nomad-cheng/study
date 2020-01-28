#include <iostream>
#include <memory>

class A
{
public:
    void classAMethod() {
        std::cout << "A::classAMethod()" << std::endl;
    }

};

int main()
{
    std::shared_ptr<A> p1(new A);
    std::cout << p1.get() << std::endl;

    std::shared_ptr<A> p2(p1);

    std::cout << p1.get() << std::endl;
    std::cout << p1.get() << std::endl;

    std::cout << p1.use_count() << std::endl;
    std::cout << p2.use_count() << std::endl;

    p1.reset();

    std::cout << "p1 after reset:" << p1.get() << std::endl;
    std::cout << p2.use_count() << std::endl;
    std::cout << p2.get() << std::endl;

    return 0;
}

