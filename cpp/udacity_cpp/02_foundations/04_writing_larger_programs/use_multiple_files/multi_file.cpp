/*
Multi-file Code
In the next few cells these functions have been separated into several different files.
The structure of the included files is as follows:
    vect_add_one --> increment_and_sum --> multiple_file

g++ -std=c++17 ./main.cpp ./increment_and_sum.cpp ./vect_add_one.cpp

*/

#include <iostream>
#include <vector>
#include "increment_and_sum.hpp"

using namespace std;

int main()
{
    vector<int> v{1, 2, 3, 4};

    int total = incrementAndComputeVectorSum(v);
    cout << "The total is:" << total << "\n";
}
