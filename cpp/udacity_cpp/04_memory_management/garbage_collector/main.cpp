#include "gc_pointer.hpp"
#include "leak_tester.hpp"

int main()
{
    Pointer<int> p = new int(19);
    p = new int(21);
    p = new int(28);

    return 0;
}
