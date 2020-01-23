#include <iostream>
#include <string>
#include <assert.h>
#include <cmath>

using namespace std;

class ParticleModel
{
public:
    void move(double v, double phi)
    {
        theta += phi;
        x += v * cos(theta);
        y += v * cos(theta);
        cout << "Move in the particle model" << endl;
    }

    void getInfo()
    {
        cout << "get info in particle model" << endl;
    }
protected:
    double x = 0;
    double y = 0;
    double theta = 0;
};

class BicycleModel: public ParticleModel
{
public:
    void move(double v, double phi)
    {
        theta += v / L * tan(phi);
        x += v * cos(theta);
        y += v * cos(theta);
        cout << "Move in Bicycle Model" << endl;
    }

    void getInfo()
    {
        cout << "get info in BicycleModel" << endl;
    }

private:
    double L;
};

class CarModel: public ParticleModel
{
public:
    double L;
};

int main()
{
    ParticleModel particle;
    BicycleModel bicycle;
    CarModel car;

    particle.getInfo();
    bicycle.getInfo();
    car.getInfo();

    particle.move(10, M_PI/9);
    bicycle.move(10, M_PI/9);
    car.move(10, M_PI/9);
}

