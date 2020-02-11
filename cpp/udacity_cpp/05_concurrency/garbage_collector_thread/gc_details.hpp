// This class defines an element that is stored in the garbage collection 
// information list.
template <class T>
class PtrDetails
{
  public:
    unsigned ref_count; // current reference count
    T *mem_ptr;         // pointer to allocated memory
    // isArray is true if mem_ptr points to an allocated array. It is false
    // otherwise. 
    bool is_array; // true if pointing to array
    // If mem_ptr is pointing to an allocated array, then arraySize contains its 
    // size
    unsigned array_size; // size of array

    // Here, mPtr points to the allocated memory. If this is an array, then size
    // specifies the size of the array.
    PtrDetails(T *mem_ptr, unsigned mem_size)
    {
        // TODO: Implement PtrDetails
        this->ref_count = 1;
        this->mem_ptr = mem_ptr;
        if (mem_size != 0)
            this->is_array = true;
        else
            this->is_array = false;
        this->array_size = mem_size;
    }
};

// Overloading operator== allows two class objects to be compared.
// This is needed by the STL list class.
template <class T>
bool operator==(const PtrDetails<T> &obj1, const PtrDetails<T> &obj2)
{
    // TODO: Implement operator==
    return (obj1.mem_ptr == obj2.mem_ptr) && (obj1.array_size == obj2.array_size);
}
