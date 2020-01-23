/*
composition: has a
inheritance: is a
Composition is a closely related alternative to inheritance. Composition 
involves constructing ("composing") classes from other classes, instead of 
inheriting traits from a parent class.
*/

#include <iostream>
#include <cmath>
#include <assert.h>

class LineSegment
{
public:
    double getLen()
    {
        return length;
    }
    void setLen(double l)
    {
        length = l;
    }

protected:
    double length;
};

class Circle
{
public:
    Circle(LineSegment &l):radius(l){};

    void setRadius(LineSegment &r)
    {
        radius = r;
    }

    double getArea()
    {
        return M_PI*pow(this->radius.getLen(), 2);
    }

private:
    LineSegment &radius;
};

int main()
{
    LineSegment radius;
    radius.setLen(3);
    Circle circle(radius);
    std::cout << "area: " << circle.getArea() << std::endl;
}
