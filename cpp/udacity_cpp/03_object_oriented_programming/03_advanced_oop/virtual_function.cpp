/* Virtual Functions
Polymorphism is one of the crucial components of OOP within C++. In this lab,
you'll use virtual methods and see their role in class interfaces and in the 
process of inheritance.

Virtual methods are declared (and possibly defined) in a base class, and are 
meant to be overridden by derived classes. This approach creates interfaces for 
your classes at the base level.

Here, you'll start with a Shape class as the base class for geometrical 2D 
entities. Geometrical shapes (closed curves) can be described by an area and the
length of their perimeter. area and perimeter_length should be methods of the 
base class interface. You'll declare each of these as virtual functions with a 
const = 0 flag to identify them as prototypes in the base class like this:
    class Shape {
        public:
            Shape() {}
            virtual double Area() const = 0;
            virtual double PerimeterLength() const = 0;
    };

We said that in the base class, virtual methods can but do not have to be 
implemented. If we delegate with instruction = 0 we are notifying compiler that
this (base) class doesn’t have virtual method implementation but every other 
derived class is required to implement this method.
*/

#include <iostream>
#include <cmath>

using namespace std;

class Shape
{
public:
    Shape() {};
    virtual double getArea() const = 0;
    virtual double getPerimeterLength() const = 0;
};

class Rectangle : public Shape
{
public:
    Rectangle(double w, double h): width(w), height(h) {};
    virtual double getArea() const override
    {
        std::cout << "Rectangle Area: " << width*height << "\n";
        return width*height;
    }
    virtual double getPerimeterLength() const override
    {
        std::cout << "Rectangle PerimeterLength: " << 2*(width + height) << "\n";
        return 2*(width+height);
    }
private:
    double width = 0;
    double height = 0;
};

class Circle : public Shape
{
public:
    Circle(double r): radius(r) {}
    virtual double getArea() const override
    {
        std::cout << "Circle Area: " << M_PI*pow(radius, 2) << "\n";
        return M_PI*pow(radius, 2);
    }
    virtual double getPerimeterLength() const override
    {
        std::cout << "Circle PerimeterLength:" << 2*M_PI*radius << endl;
        return 2*M_PI*radius;
    }

private:
    double radius = 0;
};

int main()
{
    Shape **shape_ptrs = new Shape*[2];
    shape_ptrs[0] = new Circle(12.31);
    shape_ptrs[1] = new Rectangle(10, 6);
    for (int i = 0; i < 2; i++)
    {
        std::cout << "Area: " << shape_ptrs[i]->getArea() << "\n";
        std::cout << "Perimeter: " << shape_ptrs[i]->getPerimeterLength() << "\n";
    }
}


