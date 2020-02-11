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

#include "gc_ptr.h"

/*
   Locking requirements:

   - Adding a reference requires a lock so that the garbage collector does not attempt
     to release the referenced object.

   - Incrementing root_ref_cnt only requires a lock when incrementing from 0 to 1,
     since all values !=0 are the same for the garbage collector

   - Removing a reference does not require a lock, since if the garbage collector does
     not see the change in time, an object is kept till the next run of the collector
     that could have been collected in the current run but this does not break anything.
*/

namespace mxgc {

using namespace std;

mutex gc_mutex;
thread_local gc_object *current=0;
vector<gc_object *> all_objects;
long gc_counter=1024;

gc_object::gc_object() noexcept
{
   end_=this+1;
   destructor=[](void*,void*){};
   root_ref_cnt.store(0,std::memory_order_relaxed);
   first.store(0,std::memory_order_relaxed);
}

gc_object::gc_object(void *e, void (*d)(void *,void *)) noexcept
{
   end_=e;
   destructor=d;
   root_ref_cnt.store(0,std::memory_order_relaxed);
   first.store(0,std::memory_order_relaxed);
}

gc_object::~gc_object()
{
   destructor(start(),end());
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

gc_base_ptr::gc_base_ptr(gc_object *o)
{
   type=(current && current->start()<=this && this<current->end()) ? GC_HEAP : ROOT;

   if(type==GC_HEAP) {
      if(o) {
         gc_mutex.lock(); // modifying object in GC_HEAP pointers requires a lock
         // use std::memory_order_relaxed, since we have a lock
         object.store(o,std::memory_order_relaxed);
         // add this pointer to list of pointers in the current object
         next.store(current->first.load(std::memory_order_relaxed),std::memory_order_relaxed);
         current->first.store(this,std::memory_order_relaxed);
         gc_mutex.unlock();
      } else {
         object.store(nullptr,std::memory_order_relaxed);
         next.store(current->first.load(std::memory_order_relaxed),std::memory_order_relaxed);
         std::atomic_thread_fence(std::memory_order_release); // use fence instead of lock
         current->first.store(this,std::memory_order_relaxed);
      }
   } else { // type==ROOT
      // the pointer is on stack
      object.store(o,std::memory_order_relaxed); // will not be accessed by GC
      next.store(nullptr,std::memory_order_relaxed);
      if(o) {
         // if cnt==0 or atomic exchange cnt -> cnt+1 fails, use lock
         int cnt=o->root_ref_cnt.load(std::memory_order_acquire);
         if(cnt==0 || !std::atomic_compare_exchange_strong(&o->root_ref_cnt,&cnt,cnt+1)) {
            gc_mutex.lock();
            std::atomic_fetch_add_explicit(&o->root_ref_cnt,1,std::memory_order_relaxed);
            gc_mutex.unlock();
         }
      }
   }
}


gc_base_ptr::gc_base_ptr(gc_base_ptr &&o)
{
   type=(current && current->start()<=this && this<current->end()) ? GC_HEAP : ROOT;

   gc_object *o2=o.object.load(std::memory_order_relaxed);
   if(type==GC_HEAP) {
      if(o2) {
         gc_mutex.lock(); // modifying object in GC_HEAP pointers requires a lock
         // use std::memory_order_relaxed, since we have a lock
         object.store(o2,std::memory_order_relaxed);
         // add this pointer to list of pointers in the current object
         next.store(current->first.load(std::memory_order_relaxed),std::memory_order_relaxed);
         current->first.store(this,std::memory_order_relaxed);
         gc_mutex.unlock();
      } else {
         object.store(nullptr,std::memory_order_relaxed);
         next.store(current->first.load(std::memory_order_relaxed),std::memory_order_relaxed);
         std::atomic_thread_fence(std::memory_order_release); // use fence instead of lock
         current->first.store(this,std::memory_order_relaxed);
      }
   } else {
      // the pointer is on stack
      object.store(o2,std::memory_order_relaxed); // will not be accessed by GC
      next.store(nullptr,std::memory_order_relaxed);
      if(o2) {
         if(o.type==ROOT) {
            o.object.store(nullptr,std::memory_order_relaxed); // steal object instead of incrementing the counter
         } else {
            // if cnt==0 or atomic exchange cnt -> cnt+1 fails, use lock
            int cnt=o2->root_ref_cnt.load(std::memory_order_acquire);
            if(cnt==0 || !std::atomic_compare_exchange_strong(&o2->root_ref_cnt,&cnt,cnt+1)) {
               gc_mutex.lock();
               std::atomic_fetch_add_explicit(&o2->root_ref_cnt,1,std::memory_order_relaxed);
               gc_mutex.unlock();
            }
         }
      }
   }
}

void gc_base_ptr::operator=(nullptr_t)
{
   if(type==GC_HEAP) {
      // removing a reference does not require lock
      object.store(nullptr,std::memory_order_relaxed);
   }
   else {
      gc_object *o=object.load(std::memory_order_relaxed);
      object.store(nullptr,memory_order_relaxed);
      if(o) std::atomic_fetch_add_explicit(&o->root_ref_cnt,-1,std::memory_order_relaxed);
   }
}

void gc_base_ptr::operator=(const gc_base_ptr &o)
{
   gc_object *o1=object.load(std::memory_order_relaxed); // GC does not write object, use relaxed
   gc_object *o2=o.object.load(std::memory_order_relaxed);

   if(o1==o2) return;

   if(type==GC_HEAP) {
      if(o2) {
         gc_mutex.lock();
         object.store(o2,std::memory_order_relaxed);
         gc_mutex.unlock();
      } else {
         object.store(nullptr,std::memory_order_relaxed);
      }
   }
   else {
      if(o1) std::atomic_fetch_add_explicit(&o1->root_ref_cnt,-1,std::memory_order_relaxed);
      object.store(o2,std::memory_order_relaxed); // GC does not access object
      if(o2) {
         // if cnt==0 or atomic exchange cnt -> cnt+1 fails, use lock
         int cnt=o2->root_ref_cnt.load(std::memory_order_acquire);
         if(cnt==0 || !std::atomic_compare_exchange_strong(&o2->root_ref_cnt,&cnt,cnt+1)) {
            gc_mutex.lock();
            std::atomic_fetch_add_explicit(&o2->root_ref_cnt,1,std::memory_order_relaxed);
            gc_mutex.unlock();
         }
      }
   }
}

void gc_base_ptr::operator=(gc_base_ptr &&o)
{
   gc_object *o1=object.load(std::memory_order_relaxed); // GC does not write object, use relaxed
   gc_object *o2=o.object.load(std::memory_order_relaxed);

   if(o1==o2) return;

   if(type==GC_HEAP) {
      if(o2) {
         gc_mutex.lock();
         object.store(o2,std::memory_order_relaxed);
         gc_mutex.unlock();
      } else {
         object.store(nullptr,std::memory_order_relaxed);
      }
   }
   else {
      // type==ROOT
      if(o.type==ROOT) {
         // swap pointers
         object.store(o2,std::memory_order_relaxed);
         o.object.store(o1,std::memory_order_relaxed);
      }
      else {
         if(o1) std::atomic_fetch_add_explicit(&o1->root_ref_cnt,-1,std::memory_order_relaxed);
         object.store(o2,std::memory_order_relaxed); // GC does not access object
         if(o2) {
            // if cnt==0 or atomic exchange cnt -> cnt+1 fails, use lock
            int cnt=o2->root_ref_cnt.load(std::memory_order_acquire);
            if(cnt==0 || !std::atomic_compare_exchange_strong(&o2->root_ref_cnt,&cnt,cnt+1)) {
               gc_mutex.lock();
               std::atomic_fetch_add_explicit(&o2->root_ref_cnt,1,std::memory_order_relaxed);
               gc_mutex.unlock();
            }
         }
      }
   }
}

void gc_base_ptr::reset()
{
   if(type==GC_HEAP) {
      object.store(nullptr,std::memory_order_relaxed);
   }
   else {
      gc_object *o=object.load(std::memory_order_relaxed);
      object.store(nullptr,std::memory_order_relaxed);
      if(o) std::atomic_fetch_add_explicit(&o->root_ref_cnt,-1,std::memory_order_relaxed);
   }
}

gc_base_ptr::~gc_base_ptr()
{
   if(type==ROOT) {
      gc_object *o=object.load(std::memory_order_relaxed);
      if(o) std::atomic_fetch_add_explicit(&o->root_ref_cnt,-1,std::memory_order_relaxed);
   }
}

#ifdef TIMING
timespec operator-(const timespec &a, const timespec &b)
{
   timespec r;
   r.tv_sec=a.tv_sec-b.tv_sec;
   r.tv_nsec=a.tv_nsec-b.tv_nsec;
   if(r.tv_nsec<0) { r.tv_nsec+=1000000000; r.tv_sec--; }
   return r;
}
#endif

/** run garbage collector
 */
void gc_collect()
{
#ifdef TIMING
   printf("collect: %ld\n",all_objects.size());
   timespec a,b,c,d,t;
   clock_gettime(CLOCK_MONOTONIC,&a);
#endif


   gc_mutex.lock();
   if(all_objects.empty()) {
      gc_mutex.unlock();
      return;
   }

   vector<gc_object *> pending; // list of objects to be processed next
   //pending.reserve(all_objects.size());

   // mark all objects that are referenced by pointers from the root set
   for(unsigned i=0;i<all_objects.size();i++) {
      gc_object *c=all_objects[i];

      if(c->root_ref_cnt) {
         c->mark=true;
         gc_base_ptr *first=c->first.load(std::memory_order_relaxed);
         std::atomic_thread_fence(std::memory_order_acquire);
         for(gc_base_ptr *j=first;j;j=j->next.load(std::memory_order_relaxed)) {
            // push all objects referenced by this object to the list of pending objects
            gc_object *o=j->object.load(std::memory_order_relaxed);
            if(o) pending.push_back(o);
         }
      } else {
         c->mark=false;
      }
   }

#ifdef TIMING
   clock_gettime(CLOCK_MONOTONIC,&b);
#endif

   // mark all referenced objects
   while(pending.size()) {
      gc_object *c=pending.back();
      pending.pop_back();
      if(c->mark) continue;

      // push referenced objects to list of pending objects
      c->mark=true;
      gc_base_ptr *first=c->first.load(std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_acquire);
      for(gc_base_ptr *j=first;j;j=j->next.load(std::memory_order_relaxed)) {
         gc_object *o=j->object.load(std::memory_order_relaxed);
         if(o && !o->mark) pending.push_back(o);
      }
   }

#ifdef TIMING
   clock_gettime(CLOCK_MONOTONIC,&c);
#endif

   // sort objects: first marked objects, then unmarked ones
   unsigned i,j;
   i=0;
   j=all_objects.size()-1;
   while(i<j) {
      while(i<j && all_objects[i]->mark==true) i++;
      while(i<j && all_objects[j]->mark==false) j--;
      if(i<j) {
         swap(all_objects[i],all_objects[j]);
      }
   }

   if(all_objects[i]->mark==true) i++; // may happen if no objects with mark==false

   // now i .. all_objects.size() are garbage
   pending.assign(all_objects.begin()+i,all_objects.end());
   all_objects.resize(i);
   gc_counter=2*all_objects.size();
   if(gc_counter<1024) gc_counter=1024;

   gc_mutex.unlock(); // lock must not be set when invoking destructors since these could call gc_collect()

   for(unsigned i=0;i<pending.size();i++) {
      // run destructor
      pending[i]->~gc_object();
   }

   // strangely enough having a lock in place speeds up ::operator delete()
   // therefore, set a lock although we would not need one
   gc_mutex.lock();
   for(unsigned i=0;i<pending.size();i++) {
      // and release memory
      ::operator delete(pending[i]);
   }
   gc_mutex.unlock();


#ifdef TIMING
   clock_gettime(CLOCK_MONOTONIC,&d);
   static long p1,p2,p3;
   t=b-a;
   p1+=t.tv_sec*1000000+t.tv_nsec/1000;
   t=c-b;
   p2+=t.tv_sec*1000000+t.tv_nsec/1000;
   t=d-c;
   p3+=t.tv_sec*1000000+t.tv_nsec/1000;

   printf("Phase 1: %ldms\n",p1/1000);
   printf("Phase 2: %ldms\n",p2/1000);
   printf("Phase 3: %ldms\n",p3/1000);
   printf("Total  : %ldms\n",(p1+p2+p3)/1000);
   printf("objects: %ld\n",all_objects.size());
#endif
}

} // namespace
