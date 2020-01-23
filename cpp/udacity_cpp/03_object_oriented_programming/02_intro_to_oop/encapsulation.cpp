#include <iostream>

class Encapsulation
{
public:
    void set(int a)
    {
        x = a;
    }

    int get()
    {
        return x;
    }

private:
    int x;
};


int main()
{
    Encapsulation obj;
    obj.set(5);
    std::cout << obj.get() << "\n";
}
