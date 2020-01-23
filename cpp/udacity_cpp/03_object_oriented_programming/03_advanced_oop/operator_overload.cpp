/*
Exercise: Operator Overload
Thus far, you've experimented with two types of polymorphism in C++.
These are runtime and compile time polymorphism. You've already seen how compile 
time polymorphism is achieved with function overloading.
In this lab you'll see it can be
used for operator overloading as well. In fact, you can define any operator in 
the ASCII table and give it your own set of rules!
Operator overloading can be useful for many things. Consider the + operator.
We can use it to add ints, doubles, floats, or even std::strings.
Imagine vector addition. You might want to perform vector addition on a pair of
points to add their x and y components.
The compiler won't recognize this type of
operation on its own, because this data is user defined. However, you
can overload the + operator so it performs the action that you want to implement.
*/

#include <iostream>

class Point
{
public:
    Point(int x = 0, int y = 0): x(x), y(y) {};
    Point operator - (Point const &obj)
    {
        Point res;
        res.x = x - obj.x;
        res.y = y - obj.y;
        return res;
    }
    Point operator + (Point const &obj)
    {
        Point res;
        res.x = Point::x + obj.x;
        res.y = Point::y + obj.y;
        return res;
    }
    void print()
    {
        std::cout << "Point: (" << Point::x << "," << Point::y << ")\n";
    }

private:
    int x;
    int y;
};

int main()
{
    Point p1(10, 5);
    Point p2(2, 4);
    Point p = p1+p1;
    p.print();
    p = p1 - p2;
    p.print();
}

