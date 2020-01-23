#include <iostream>
#include <string>

using namespace std;

class Abstraction
{
public:
   void setAttributes(int number, int character)
   {
        this->number = number;
        this->character = character;
   }
   void getAttributes()
   {
       processAttributes();
       cout << "number: " << number << " character: " << character << endl;
   }
private:
    void processAttributes()
    {
        this->number *= 6;
        this->character += 1;
    }
    int number;
    int character;
};

int main()
{
    Abstraction ab;
    ab.setAttributes(1, 2);
    ab.getAttributes();
}
