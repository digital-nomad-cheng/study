#include <iostream>
#include <cmath>

void trigonometry(double angle, double *_sin, double *_cos)
{
    *_sin = std::sin(angle);
    *_cos = std::cos(angle);
}

int main()
{
    double angle, _sin, _cos;
    std::cout << "Write angle in radians:";
    std::cin >> angle;
    std::cout << "Trigonometry values are: " << std::endl;
    trigonometry(angle, &_sin, &_cos);
    std::cout << "Sine is: " << _sin << std::endl;
    std::cout << "Cosine is: " << _cos << std::endl;
    
    return 0;
}

