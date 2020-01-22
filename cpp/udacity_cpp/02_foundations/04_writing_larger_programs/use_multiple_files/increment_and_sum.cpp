#include "vector_add_one.hpp"

int incrementAndComputeVectorSum(vector<int> v)
{
    int total = 0;
    addOneToEach(v);

    for (auto i: v) {
        total += i;
    }
    return total;
}
