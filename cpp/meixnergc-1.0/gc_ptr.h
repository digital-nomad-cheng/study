/*
Copyright (c) 2017 Dr. Matthias Meixner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef GC_PTR_H
#define GC_PTR_H

/**  \file gc_ptr.h */

#include <mutex>
#include <vector>
#include <cstddef>
#include <atomic>

#ifndef DOXYGEN
namespace mxgc {
#endif

class gc_base_ptr;
class gc_object;

/** manually trigger garbage collection.
 * Normally this is not required as the collection starts automatically.
 * It may be useful in low-memory situations to force a garbage collection.
 */
 void gc_collect();

extern std::mutex gc_mutex;                  /**< global mutex */
extern thread_local gc_object *current;      /**< pointer to current object under construction in current thread */
extern std::vector<gc_object *> all_objects; /**< array containing all allocated objects */
extern long gc_counter;                      /**< when this counter reaches 0, the garbage collection is triggered */

/** GC memory control object. All allocations are handled via gc_object and C++ objects are constructed within them */
class gc_object {

   template<class T> friend class make_gc;
   friend class gc_base_ptr;
   friend void gc_collect();

 private:
   gc_object(const gc_object &)=delete;
   void operator=(const gc_object &)=delete;

 protected:
   void *end_; /**< End of the memory range allocated by this object */

   /** determine the start address of the first contained C++ object belonging to this allocation:
    * It starts after the control information contained in gc_object, i.e. the layout in memory
    * is as follows: \<gc_obj>\<C++ object>[\<C++ object> ....]
    * \return start address of first contained object.
    */
   void *start() noexcept { return (void *)(this+1); }

   /** determine the end address of the last contaned C++ object belonging to this allocation
    * \return end address after last contained object.
    */
   void *end() noexcept { return end_; }

   /** destructor callback that destructs the contained elements.
    * \param[in] s start address
    * \param[in] e end addresses */
   void (*destructor)(void *s, void *e);

   /** number of pointers from the root set that point to this object */
   std::atomic<int> root_ref_cnt;

   /** pointer to first contained gc_ptr<> within this object */
   std::atomic<gc_base_ptr *> first;

   /** flag for mark-sweek algorithm */
   bool mark;

 public:

   /** constructor */
   gc_object() noexcept;

   /** constructor
    * \param[in] e
    * \param[in] d destructor callback
    */
   gc_object(void *e,void (*d)(void *,void *)) noexcept;

   /** destructor */
   ~gc_object();
};

/** base class for gc_ptr containing all shared code */
class gc_base_ptr {

   friend void gc_collect();

   static void gc_collect(gc_object *o);

 protected:
   /** pointer type */
   enum PtrType {
      ROOT,    /**< pointer belongs to the root set */
      GC_HEAP  /**< pointer resides in an allocated object */
   } type; /**< pointer type */

   std::atomic<gc_base_ptr *> next;       /**< pointer to next gc_ptr within the same allocated object */
   std::atomic<gc_object *>   object;     /**< pointer to the object */

 public:
   /** constructor */
   explicit gc_base_ptr(gc_object *c=nullptr);

   /** copy constructor */
   gc_base_ptr(const gc_base_ptr &o) : gc_base_ptr(o.object) {};

   /** move constructor */
   gc_base_ptr(gc_base_ptr &&o);

   /** assignment operator */
   void operator=(nullptr_t);

   /** assignment operator */
   void operator=(const gc_base_ptr &o);

   /** move assignment operator */
   void operator=(gc_base_ptr &&o);

   /** reset to nullptr */
   void reset();

   /** destructor */
   ~gc_base_ptr();
};


/** garbage collected pointer */
template<class T> class gc_ptr: public gc_base_ptr {
   template<class U> friend class gc_ptr;
 protected:
   T *ptr; /**< managed pointer */
 public:


   /** constructor, set up gc_ptr with NULL */
   gc_ptr():gc_base_ptr() { ptr=nullptr;  }

   /** constructor, set up gc_ptr with NULL */
   gc_ptr(nullptr_t):gc_base_ptr() { ptr=nullptr; }

   /** copy constructor
    * \param[in] o pointer to be copied
    */
   gc_ptr(const gc_ptr &o):gc_base_ptr(o) {  ptr=o.ptr; }

   /** move constructor
    * \param[in] o pointer to be moved
    */
   gc_ptr(gc_ptr &&o):gc_base_ptr(std::move(o)) {  ptr=o.ptr; }

   /** aliasing constructor, the pointer \a p is stored, but references to \a o are set up.
    * This is useful, when the pointer points to an inner element of \a o
    * \param[in] o referenced object, used for garbage collection
    * \param[in] p pointer
    */
   gc_ptr(const gc_base_ptr &o, T *p):gc_base_ptr(o) { ptr=p; }


   /** cast constructor
    * \param[in] o pointer to be copied
    */
   template<class U> gc_ptr(const gc_ptr<U> &o):gc_base_ptr(o) {
      ptr=o.ptr;
   }


   /** assignment operator, emulates standard C++ casting rules
    * \param[in] o pointer to be assigned.
    */
   gc_ptr &operator=(const gc_ptr &o) {
      gc_base_ptr::operator=(o);
      ptr=o.ptr;
      return *this;
   }

   /** move operator, emulates standard C++ casting rules
    * \param[in] o pointer to be assigned.
    */
   gc_ptr &operator=(gc_ptr &&o) {
      gc_base_ptr::operator=(std::move(o));
      ptr=o.ptr;
      return *this;
   }

   /** assignment with nullptr */
   gc_ptr &operator=(nullptr_t) {
      gc_base_ptr::operator=(nullptr);
      ptr=nullptr;
      return *this;
   }



   /** get pointer \return pointer */
   T *get() const noexcept {return ptr;}

   /** get pointer \return pointer */
   operator T*() const noexcept { return ptr; }

   /** dereference pointer */
   T &operator*() const noexcept {return *ptr;}

   /** access via pointer */
   T *operator->() const noexcept {return ptr;}

   /** array access */
   T &operator[](int idx) const noexcept { return ptr[idx]; }

   /** reset pointer to null */
   void reset() { ptr=nullptr; gc_base_ptr::reset(); }

   /** prefix increment */
   gc_ptr<T> &operator++() noexcept { ++ptr; return *this; }

   /** prefix decrement */
   gc_ptr<T> &operator--() noexcept { --ptr; return *this; }

   /** postfix increment */
   gc_ptr<T> operator++(int) noexcept { gc_ptr<T> r(*this); ptr++; return r; }

   /** postfix decrement */
   gc_ptr<T> operator--(int) noexcept { gc_ptr<T> r(*this); ptr--; return r; }


   /** += operator */
   template<class U> gc_ptr<T> &operator+=(const U &z) noexcept { ptr+=z; return *this; }

   /** -= operator */
   template<class U> gc_ptr<T> &operator-=(const U &z) noexcept { ptr-=z; return *this; }
};


/** class for allocating new objects */
template<class T> class make_gc: public gc_ptr<T>
{
 public:
   /** Allocate object and return gc_ptr to it.
    * \tparam T object type to allocate
    * \param[in] args arguments that are forwarded to the constructor of the object
    */
   template <class... Args> make_gc(Args&&... args)
   {
      // allocate memory for control object and T object
      char *mem=(char*)::operator new(sizeof(gc_object)+sizeof(T));

      // creating the control object and pushing it on the stack requires a lock
      gc_mutex.lock();

      if(gc_counter--==0) {
         gc_mutex.unlock();
         gc_collect();
         gc_mutex.lock();
      }

      // construct control object in allocated memory
      gc_object *new_object=new(mem) gc_object(mem+sizeof(gc_object)+sizeof(T),
                                             [](void *s, void *e) { T *t=(T*)s; t->~T(); });
      gc_base_ptr::object.store(new_object,std::memory_order_relaxed);


      // increment reference count if needed
      if(gc_base_ptr::type==gc_base_ptr::ROOT) new_object->root_ref_cnt.store(1,std::memory_order_relaxed);

      // push object on object list
      all_objects.push_back(new_object);

      gc_mutex.unlock();

      // store old current object in case we are in a recursion
      // and set up new current object
      gc_object *parent=current;
      current=new_object;

      // set up pointer
      gc_ptr<T>::ptr=(T*)new_object->start();

      // run the constructor of the allocated object
      try {
         new(new_object->start()) T(std::forward<Args>(args)...);
      }
      catch(...) {
         // if it fails, release reference, so that GC will clean it up
         if(gc_base_ptr::type==gc_base_ptr::ROOT) new_object->root_ref_cnt--;
         gc_base_ptr::object=nullptr;
         gc_ptr<T>::ptr=nullptr;
         current=parent;
         throw; // re-throw exception
      }

      current=parent;
   }
};

/** class for allocating new arrays of objects */
template<class T> class make_gc<T[]>: public gc_ptr<T>
{
 public:
   /** Allocate array of objects
    * \tparam T object type to allocate
    * \param[in] size array size
    */
   make_gc(unsigned size)
   {
      // allocate memory for control object and T object
      char *mem=(char*)::operator new(sizeof(gc_object)+size*sizeof(T));

      // creating the control object and pushing it on the stack requires a lock
      gc_mutex.lock();

      if(gc_counter--==0) {
         gc_mutex.unlock();
         gc_collect();
         gc_mutex.lock();
      }

      gc_object *new_object=new(mem) gc_object(mem+sizeof(gc_object)+size*sizeof(T),
                                             [](void *s, void *e) { for(T *t=(T*)e;--t>=s;) t->~T(); });
      gc_base_ptr::object.store(new_object,std::memory_order_relaxed);


      // increment reference count if needed
      if(gc_base_ptr::type==gc_base_ptr::ROOT) new_object->root_ref_cnt.store(1,std::memory_order_relaxed);

      // push object on object list
      all_objects.push_back(new_object);

      gc_mutex.unlock();

      // store old current object in case we are in a recursion
      // and set up new current object
      gc_object *parent=current;
      current=new_object;

      T *start=(T*)new_object->start();
      T *end=(T*)new_object->end();

      gc_ptr<T>::ptr=start;
      T *i;
      // run the constructor of the allocated object
      try {
         for(i=start;i!=end;i++) new(i) T();
      }
      catch(...) {
         for(T *j=i;--j>=start;) j->~T();
         if(gc_base_ptr::type==gc_base_ptr::ROOT) new_object->root_ref_cnt--;
         gc_base_ptr::object=nullptr;
         gc_ptr<T>::ptr=nullptr;
         current=parent;
         throw; // re-throw exception
      }

      current=parent;
   }
};

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
///////////////// gc_ptr operators ///////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////

// comparison operators

/** compare pointers
 * \param[in] lhs first pointer
 * \param[in] rhs second pointer
 * \return true lhs.get()==rhs.get()
 */
template <class T, class U>  bool operator==(const gc_ptr<T>& lhs, const gc_ptr<U>& rhs) noexcept
{
   return lhs.get()==rhs.get();
}

/** compare pointers
 * \param[in] lhs first pointer
 * \param[in] n nullptr
 * \return true lhs.get()==0
 */
template <class T> bool operator==(const gc_ptr<T>& lhs, nullptr_t n) noexcept
{
   return lhs.get()==0;
}

/** compare pointers
 * \param[in] n nullptr
 * \param[in] rhs second pointer
 * \return true 0==rhs.get()
 */
template <class T> bool operator==(nullptr_t n, const gc_ptr<T>& rhs) noexcept
{
   return 0==rhs.get();
}

/** compare pointers
 * \param[in] lhs first pointer
 * \param[in] rhs second pointer
 * \return true lhs.get()!=rhs.get()
 */
template <class T, class U>  bool operator!=(const gc_ptr<T>& lhs, const gc_ptr<U>& rhs) noexcept
{
   return lhs.get()!=rhs.get();
}

/** compare pointers
 * \param[in] lhs first pointer
 * \param[in] n nullptr
 * \return true lhs.get()!=0
 */
template <class T> bool operator!=(const gc_ptr<T>& lhs, nullptr_t n) noexcept
{
   return lhs.get()!=0;
}

/** compare pointers
 * \param[in] n nullptr
 * \param[in] rhs second pointer
 * \return true 0!=rhs.get()
 */
template <class T> bool operator!=(nullptr_t n, const gc_ptr<T>& rhs) noexcept
{
   return 0!=rhs.get();
}

/** compare pointers
 * \param[in] lhs first pointer
 * \param[in] rhs second pointer
 * \return true lhs.get()<rhs.get()
 */
template <class T, class U> bool operator<(const gc_ptr<T>& lhs, const gc_ptr<U>& rhs) noexcept
{
   return lhs.get()<rhs.get();
}

/** compare pointers
 * \param[in] lhs first pointer
 * \param[in] n nullptr
 * \return true lhs.get()<0
 */
template <class T> bool operator<(const gc_ptr<T>& lhs, nullptr_t n) noexcept
{
   return lhs.get()<0;
}

/** compare pointers
 * \param[in] n nullptr
 * \param[in] rhs second pointer
 * \return true 0<rhs.get()
 */
template <class T> bool operator<(nullptr_t n, const gc_ptr<T>& rhs) noexcept
{
   return 0<rhs.get();
}

/** compare pointers
 * \param[in] lhs first pointer
 * \param[in] rhs second pointer
 * \return true lhs.get()<=rhs.get()
 */
template <class T> bool operator<=(const gc_ptr<T>& lhs, const gc_ptr<T>& rhs) noexcept
{
   return lhs.get()<=rhs.get();
}

/** compare pointers
 * \param[in] lhs first pointer
 * \param[in] n nullptr
 * \return true lhs.get()<=0
 */
template <class T> bool operator<=(const gc_ptr<T>& lhs, nullptr_t n) noexcept
{
   return lhs.get()<=0;
}

/** compare pointers
 * \param[in] n nullptr
 * \param[in] rhs second pointer
 * \return true 0<=rhs.get()
 */
template <class T> bool operator<=(nullptr_t n, const gc_ptr<T>& rhs) noexcept
{
   return 0<=rhs.get();
}

/** compare pointers
 * \param[in] lhs first pointer
 * \param[in] rhs second pointer
 * \return true lhs.get()>rhs.get()
 */
template <class T> bool operator>(const gc_ptr<T>& lhs, const gc_ptr<T>& rhs) noexcept
{
   return lhs.get()>rhs.get();
}

/** compare pointers
 * \param[in] lhs first pointer
 * \param[in] n nullptr
 * \return true lhs.get()>0
 */
template <class T> bool operator>(const gc_ptr<T>& lhs, nullptr_t n) noexcept
{
   return lhs.get()>0;
}

/** compare pointers
 * \param[in] n nullptr
 * \param[in] rhs second pointer
 * \return true 0>rhs.get()
 */
template <class T> bool operator>(nullptr_t n, const gc_ptr<T>& rhs) noexcept
{
   return 0>rhs.get();
}

/** compare pointers
 * \param[in] lhs first pointer
 * \param[in] rhs second pointer
 * \return true lhs.get()>=rhs.get()
 */
template <class T> bool operator>=(const gc_ptr<T>& lhs,  const gc_ptr<T>& rhs) noexcept
{
   return lhs.get()>=rhs.get();
}

/** compare pointers
 * \param[in] lhs first pointer
 * \param[in] n nullptr
 * \return true lhs.get()>=0
 */
template <class T> bool operator>=(const gc_ptr<T>& lhs, nullptr_t n) noexcept
{
   return lhs.get()>=0;
}

/** compare pointers
 * \param[in] n nullptr
 * \param[in] rhs second pointer
 * \return true 0>=rhs.get()
 */
template <class T> bool operator>=(nullptr_t n, const gc_ptr<T>& rhs) noexcept
{
   return 0>=rhs.get();
}


/** pointer arithmetic
 * \param[in] a pointer
 * \param[in] b integer type
 * \return resulting pointer a+u
 */
template<class T, class U> gc_ptr<T> operator+(const gc_ptr<T> &a, const U &b) noexcept
{
   return gc_ptr<T>(a,a.get()+b);
}

/** pointer arithmetic
 * \param[in] a integer type
 * \param[in] b pointer
 * \return resulting pointer a+u
 */
template<class T, class U> gc_ptr<T> operator+(const U &a, const gc_ptr<T> &b) noexcept
{
   return gc_ptr<T>(b,b.get()+a);
}

/** pointer arithmetic
 * \param[in] a pointer
 * \param[in] b integer type
 * \return resulting pointer a-u
 */
template<class T, class U> gc_ptr<T> operator-(const gc_ptr<T> &a, const U &b) noexcept
{
   return gc_ptr<T>(a,a.get()-b);
}

/** pointer arithmetic
 * \param[in] a pointer
 * \param[in] b pointer
 * \return resulting integer value a-b
 */
template<class T>  auto operator-(const gc_ptr<T> &a, const gc_ptr<T> &b) noexcept -> decltype(a.get()-b.get())
{
   return a.get()-b.get();
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
///////////////// gc_ptr cast ///////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////

/** gc_ptr version of static_cast<>()
 * \return gc_ptr as result of static_cast()
 */
template <class T, class U>  gc_ptr<T> static_pointer_cast(const gc_ptr<U>& p) noexcept
{
   return gc_ptr<T>(p,static_cast<T*>(p.get()));
}

/** gc_ptr version of dynamic_cast<>()
 * \return gc_ptr as result of static_cast()
 */
template <class T, class U>  gc_ptr<T> dynamic_pointer_cast(const gc_ptr<U>& p) noexcept
{
   return gc_ptr<T>(p,dynamic_cast<T*>(p.get()));
}

/** gc_ptr version of const_cast<>()
 * \return gc_ptr as result of static_cast()
 */
template <class T, class U>  gc_ptr<T> const_pointer_cast(const gc_ptr<U>& p) noexcept
{
   return gc_ptr<T>(p,const_cast<T*>(p.get()));
}

/** gc_ptr version of reinterpret_cast<>()
 * \return gc_ptr as result of static_cast()
 */
template <class T, class U>  gc_ptr<T> reinterpret_pointer_cast(const gc_ptr<U>& p) noexcept
{
   return gc_ptr<T>(p,reinterpret_cast<T*>(p.get()));
}

/** gc_ptr version of C-like cast
 * \return gc_ptr as result of C-like cast
 */
template <class T, class U>  gc_ptr<T> pointer_cast(const gc_ptr<U>& p) noexcept
{
   return gc_ptr<T>(p,(T*)(p.get()));
}

#ifndef DOXYGEN
} // namespace
#endif

#endif
