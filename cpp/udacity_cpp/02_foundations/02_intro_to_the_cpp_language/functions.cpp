#include <iostream>
#include <vector>

using std::cout;
using std::string;
using std::vector;

int additionFunction(int i, int j)
{
    return i + j;
}


void printStrings(string a, string b)
{
    cout << a << " " << b << "\n";
}

int vectorAdditionFunction(vector<int> v)
{
    int sum = 0;
    for (int num: v) {
        sum += num;
    }
    return sum;
}

int main()
{
    // test additionFunction
    auto d = 3;
    auto f = 7;
    cout << additionFunction(d, f) << "\n";
    
    // test printStrings
    string s1 = "C++ is";
    string s2 = "super awesome";
    printStrings(s1, s2);
    
    // test vectorAdditionFunction
    vector<int> v {1, 2, 3};
    cout << vectorAdditionFunction(v) << "\n"; 
}
