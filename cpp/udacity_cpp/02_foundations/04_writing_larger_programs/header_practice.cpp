#include <iostream>
#include <vector>
#include "header_practice.hpp"

using namespace std;

int incrementAndComputeVectorSum(vector<int> v)
{
    int total = 0;
    addOneToEach(v);
    for (auto i: v) {
        total += i;
    }
    return total;
}

void addOneToEach(vector<int> &v) 
{

    // Note that the function passes a reference to v
    // and the for loop below uses references to
    // each item in v. This means the actual
    // ints that v holds will be incremented.
    for (auto &i: v) {
        i++;
    }
}

int main()
{
    vector<int> v{1, 2, 3, 4};
    int total = incrementAndComputeVectorSum(v);
    cout << "The total is:" << total << "\n";
    return 0;
}
