/* [description of Malloc()]
 * Malloc is a function which is essentially forerunner to new operator in C++, 
 * originally inherited from C. Its function is to allocate a block of specific 
 * size, measured in bytes of memory, where result of this operation (function) 
 * is pointer to the beginning of the block. Data stored in this memory block is
 * not initialized, which means that this space has non determinate values.
 * Prototype of this function is:
 * void* malloc(size_t size);
 * Size is unsigned integer value which represent memory block in bytes. 
 * It is important to note that in case of failed allocation return value is 
 * null pointer.
 */

#include <iostream>
#include <cstring>
#include <cstdlib>

int main()
{
    int *ptr = NULL;
    ptr = new(std::nothrow) int;
    ptr = (int *)std::malloc(4*sizeof(int));
    for (int i = 0; i < 4; i++) {
        std::cout << "Address: " << &ptr[i] << " Value 
        " << ptr[i] << std::endl;
    }
    
    for (int i = 0; i < 4; i++) {
        std::memset(&ptr[i], ('A'+i), sizeof(int));
    }

    for (int i = 0; i < 4; i++) {
        std::cout << "Address: " << &ptr[i] << " Value: " << char(ptr[i])
        << std::endl;
    }

    return 0;
}


