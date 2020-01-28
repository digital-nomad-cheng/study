#include <iostream>
#include <list>
#include <typeinfo>
#include <cstdlib>
#include "gc_details.hpp"
#include "gc_iterator.hpp"

#define DISPLAY true

/*Pointer implements a pointer type that uses garbage collection to release 
 *unused memory. A Pointer must only be used to point to memory that was 
 *dynamically allocated using new. When used to refer to an allocated array,
 *specify the array size.
 */
template <class T, int size = 0>
class Pointer{
private:
    // ref_container maintains the garbage collection list.
    static std::list<PtrDetails<T> > ref_container;
    // addr points to the allocated memory to which
    // this Pointer pointer currently points.
    T *addr;
    // is_array is true if this Pointer points to an allocated array. 
    // It is false otherwise.
    bool is_array;
    // true if pointing to array If this Pointer is pointing to an allocated
    // array, then array_size contains its size.
    unsigned array_size; // size of the array
    static bool first; // true when first Pointer is created
    // Return an iterator to pointer details in ref_container.
    typename std::list<PtrDetails<T> >::iterator findPtrInfo(T *ptr);
public:
    // Define an iterator type for Pointer<T>.
    typedef Iter<T> GC_iterator;
    // Empty constructor
    // NOTE: templates aren't able to have prototypes with default arguments
    // this is why constructor is designed like this:
    Pointer(){
        Pointer(NULL);
    }
    Pointer(T*);
    // Copy constructor.
    Pointer(const Pointer &);
    // Destructor for Pointer.
    ~Pointer();
    // Collect garbage. Returns true if at least
    // one object was freed.
    static bool collect();
    // Overload assignment of pointer to Pointer.
    T *operator=(T *t);
    // Overload assignment of Pointer to Pointer.
    Pointer &operator=(Pointer &rv);
    // Return a reference to the object pointed
    // to by this Pointer.
    T &operator*(){
        return *addr;
    }
    // Return the address being pointed to.
    T *operator->() { return addr; }
    // Return a reference to the object at the
    // index specified by i.
    T &operator[](int i){ return addr[i];}
    // Conversion function to T *.
    operator T *() { return addr; }
    // Return an Iter to the start of the allocated memory.
    Iter<T> begin(){
        int _size;
        if (is_array)
            _size = array_size;
        else
            _size = 1;
        return Iter<T>(addr, addr, addr + _size);
    }
    // Return an Iter to one past the end of an allocated array.
    Iter<T> end(){
        int _size;
        if (is_array)
            _size = array_size;
        else
            _size = 1;
        return Iter<T>(addr + _size, addr, addr + _size);
    }
    // Return the size of ref_container for this type of Pointer.
    static int ref_container_size() { return ref_container.size(); }
    // A utility function that displays ref_container.
    static void showList();
    // Clear ref_container when program exits.
    static void shutDown();
};

// STATIC INITIALIZATION
// Creates storage for the static variables
template <class T, int size>
std::list<PtrDetails<T> > Pointer<T, size>::ref_container;
template <class T, int size>
bool Pointer<T, size>::first = true; // used to register a shut down function 
                                     // when the program ends

// Constructor for both initialized and uninitialized objects. 
// see class interface
template<class T, int size>
Pointer<T, size>::Pointer(T *t){
    // Register shutDown() as an exit function.
    if (first)
        atexit(shutDown);
    first = false;
    // TODO: Implement Pointer constructor
    // Lab: Smart Pointer Project Lab
    typename std::list<PtrDetails<T>>::iterator p;
    p = findPtrInfo(t);
    if (p != ref_container.end())
        p->ref_count++;
    else {
        PtrDetails<T> obj(t, size);
        ref_container.push_front(obj);
    }
    
    addr = t;
    array_size = size;
    if (size > 0 )
        is_array = true;
    else
        is_array = false;
    
    #ifdef DISPLAY
        std::cout << "Pointer Constructor";
        if (is_array) 
            std::cout << " Size is " << array_size << std::endl;
        else
            std::cout << std::endl;
    #endif
}

// Copy constructor.
template< class T, int size>
Pointer<T, size>::Pointer(const Pointer &obj){

    // TODO: Implement Pointer constructor
    // Lab: Smart Pointer Project Lab
    typename std::list<PtrDetails<T>>::iterator p;
    p = findPtrInfo(obj.addr);
    // Todo what if out of range
    if (p != ref_container.end())
        p->ref_count++;
    else {
        PtrDetails<T> ptr_detail(obj.addr, obj.array_size);
        ref_container.push_front(ptr_detail);
    }

    addr = obj.addr;
    array_size = obj.array_size;
    if (array_size > 0)
        is_array = true;
    else
        is_array = false;

    #ifdef DISPLAY
        std::cout << "Consturcting copy.";
        if (is_array)
            std::cout << " Size is " << array_size << std::endl;
        else
            std::cout << std::endl;
    #endif
}

// Destructor for Pointer.
template <class T, int size>
Pointer<T, size>::~Pointer(){

    // TODO: Implement Pointer destructor
    // Lab: New and Delete Project Lab
    typename std::list<PtrDetails<T>>::iterator p;
    p = findPtrInfo(addr);
    if (p->ref_count)
        p->ref_count--;
    #ifdef DISPLAY
        std::cout << "Pointer (w/ garbage collection) going out of scope.\n";
    #endif
    collect();


}

// Collect garbage. Returns true if at least
// one object was freed.
template <class T, int size>
bool Pointer<T, size>::collect(){

    // TODO: Implement collect function
    // LAB: New and Delete Project Lab
    // Note: collect() will be called in the destructor
    if (ref_container_size() == 0)
        return false; // list is empty
    bool collected = false;
    typename std::list<PtrDetails<T> >::iterator p;
    // do {

    //     // Scan ref_container looking for unreferenced pointers.
    //     for (p = ref_container.begin(); p != ref_container.end(); p++) {
    //         // If in-use, skip.
    //         if (p->ref_count > 0) continue;

    //         collected = true;

    //         // Remove unused entry from ref_container.
    //         ref_container.remove(*p);

    //         // Free memory unless the Pointer is null.
    //         if (p->mem_ptr) {
    //             if (p->is_array) {
    //                 #ifdef DISPLAY
    //                     std::cout << "Deleting array of size "
    //                               << p->array_size << std::endl;
    //                 #endif
    //                 delete[] p->mem_ptr; // delete array
    //             }
    //             else {
    //                 #ifdef DISPLAY
    //                     std::cout << "Deleting: "
    //                               << *(T *) p->mem_ptr << std::endl;
    //                 #endif
    //                 delete p->mem_ptr;    // delete signle element 
    //             }
    //         }

    //         // Restart the search.
    //         break;
    //     }

    // } while ( p != ref_container.end() );

    typename std::list<PtrDetails<T>> ptrs_to_remove;

    for (p = ref_container.begin(); p != ref_container.end(); p++)
    {
        if (p->ref_count == 0) {
            // ref_container.remove(*p);
            ptrs_to_remove.push_back(*p);
            if (p->is_array)  {
                #ifdef DISPLAY
                    std::cout << "Deleting array of size "
                        << p->array_size << std::endl;
                #endif
                delete [] p->mem_ptr;
            } else {
                #ifdef DISPLAY
                    std::cout << "Deleting: "
                        << *(T *) p->mem_ptr << std::endl;
                #endif
                delete p->mem_ptr;
                #ifdef DISPLAY
                    std::cout << "Deleted!" << std::endl;
                #endif
            }
            collected = true;
        }
    }
    for (auto &a: ptrs_to_remove) {
        ref_container.remove(a);
    }
    #ifdef DISPLAY
        std::cout << "After garbage collection for ";
        showList();
    #endif
    return collected;
}

// Overload assignment of pointer to Pointer.
template <class T, int size>
T *Pointer<T, size>::operator=(T *t){
    // TODO: Implement operator=
    // LAB: Smart Pointer Project Lab
    typename std::list<PtrDetails<T>>::iterator p;
    p = findPtrInfo(addr);
    p->ref_count--;
    p = findPtrInfo(t);
    if (p != ref_container.end())
        p->ref_count++;
    else {
        PtrDetails<T> obj(t, size);
        ref_container.push_front(obj);
    }
    addr = t;
    return t;
}

// Overload assignment of Pointer to Pointer.
template <class T, int size>
Pointer<T, size> &Pointer<T, size>::operator=(Pointer &rv)
{
    // TODO: Implement operator=
    // LAB: Smart Pointer Project Lab
    typename std::list<PtrDetails<T>>::iterator p;
    p = findPtrInfo(addr);
    p->ref_count--;

    p = findPtrInfo(rv.addr);
    p->ref_count++;

    addr = rv.addr;
    return rv;

}

// A utility function that displays ref_container.
template <class T, int size>
void Pointer<T, size>::showList(){
    typename std::list<PtrDetails<T> >::iterator p;
    std::cout << "ref_container<" << typeid(T).name() << ", " << size << ">:\n";
    std::cout << "mem_ptr ref_count value\n ";
    if (ref_container.begin() == ref_container.end()) {
        std::cout << " Container is empty!\n\n ";
    }
    for (p = ref_container.begin(); p != ref_container.end(); p++) {
        std::cout << "[" << (void *)p->mem_ptr << "]"
             << " " << p->ref_count << " ";
        if (p->mem_ptr)
            std::cout << " " << *p->mem_ptr;
        else
            std::cout << "---";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Find a pointer in ref_container.
template <class T, int size>
typename std::list<PtrDetails<T> >::iterator
Pointer<T, size>::findPtrInfo(T *ptr){
    typename std::list<PtrDetails<T> >::iterator p;
    // Find ptr in ref_container.
    for (p = ref_container.begin(); p != ref_container.end(); p++)
        if (p->mem_ptr == ptr)
            return p;
    return p;
}

// Clear ref_container when program exits. Release any memory that was prevented
// from being released because of a cicular reference
template <class T, int size>
void Pointer<T, size>::shutDown(){
    if (ref_container_size() == 0)
        return; // list is empty
    typename std::list<PtrDetails<T> >::iterator p;
    for (p = ref_container.begin(); p != ref_container.end(); p++)
    {
        // Set all reference counts to zero
        p->ref_count = 0;
    }
    collect();
}
