#include <iostream>
#include <string>

using namespace std;

class Abstraction
{
public:
    void static printCharNumber(char c)
    {
        int result = c;
        cout << result << endl;
    }
private:
    int number;
    char _char;
};

int main()
{
    char c = 'X';
    Abstraction::printCharNumber(c);
}
