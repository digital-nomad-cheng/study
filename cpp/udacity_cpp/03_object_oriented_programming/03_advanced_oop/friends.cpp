#include <iostream>
#include <string>
#include <assert.h>

class Rectangle;

class Square
{
public:
    Square(int s): side(s){};
    void test()
    {
        std::cout << side << std::endl;
    }
private:
    friend class Rectangle;
    int side;
};

class Rectangle
{
public:
    Rectangle(Square &s):width(s.side), height(s.side) {};

    float getArea() const
    {
        return width*height;
    }

private:
    float width;
    float height;
};

int main()
{
    Square s(4);
    s.test();

    Rectangle rect(s);
    assert(rect.getArea() == 16);
}
