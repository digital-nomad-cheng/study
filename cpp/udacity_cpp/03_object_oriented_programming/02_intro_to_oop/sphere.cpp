#include <iostream>
#include <string>
#include <math.h>

using namespace std;

class Sphere
{
public:
    Sphere(float r): radius(r) {};
    float getVolume()
    {
        return (4/3.0)*M_PI*radius*radius*radius;
    }
private:
    float radius;
};

int main()
{
    Sphere s(10);
    cout << "volume: " << s.getVolume() << endl;
}


