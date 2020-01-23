#include <assert.h>
#include <string>

class Person
{
public:
    Person(std::string s): name(s){}
    std::string name;
};

int main()
{
    Person alice("Alice");
    Person bob("Bob");
    assert(alice.name != bob.name);
}

