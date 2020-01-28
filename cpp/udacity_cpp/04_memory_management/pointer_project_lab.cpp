tempate <typename T>
class PtrDetails
{
public:
    int ref_count_;
    T *mem_ptr;
    int array_size_;
    PtrDetails(T *obj, int size);
};

template <class T> bool operator==(const PtrDetails<T> &obj1, 
                                   const PtrDetails<T> &obj2)
{
    return (obj1.array_size_ == obj2.array_size_ && obj1.mem_ptr == obj2.mem_ptr);
}

