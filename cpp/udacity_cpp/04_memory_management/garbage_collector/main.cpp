#include "gc_pointer.hpp"
#include "leak_tester.hpp"

int main()
{
    Pointer<int> p = new int(19);
    
    p.showList();
    p = new int(21);
    p.showList();
    p = new int(28);
    p.showList();
    
    Pointer<int> q = new int(9);
    std::cout << "q.showList(): " << std::endl;
    q.showList();
    p = q;
    std::cout << "p.showList(): " << std::endl;
    p.showList();
    return 0;
}
