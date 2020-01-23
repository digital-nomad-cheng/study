#include <iostream>
#include <string>

using namespace std;

class Abstraction
{
public:
    Abstraction()
    {
        counter++;
    }
    void setAttributes(int n, char c)
    {
        this->number = n;
        this->_char = c;
    }
    int getCounter()
    {
        return this->counter;
    }
private:
    void processAttributes();
    static int counter;
    int number;
    char _char;
};

int Abstraction::counter = 0;

int main()
{
    Abstraction ab1;
    cout << "counter: " << ab1.getCounter() << endl;
    Abstraction ab2;
    cout << "counter: " << ab2.getCounter() << endl;
}
