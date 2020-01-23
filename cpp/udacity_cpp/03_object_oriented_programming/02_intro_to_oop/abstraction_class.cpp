#include <iostream>
#include <string>

using namespace std;

class Abstraction
{
public:
    void setValue(int v1, int v2)
    {
        this->value1 = v1;
        this->value2 = v2;
    }
    void printValue() const
    {
        cout << "v1: " << value1 << "v2: " << value2 << endl;
    }
private:
    int value1;
    int value2;

};

int main()
{
    Abstraction ab;
    ab.setValue(1, 2);
    ab.printValue();
}
