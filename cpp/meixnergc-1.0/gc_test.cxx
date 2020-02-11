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

#include <time.h>
#include <stdio.h>
#include <string.h>

#include <vector>
#include <assert.h>
#include <memory>
#include <atomic>


using namespace mxgc;

enum {
   CNT=1000000,
   THREADS=10
};

using namespace std;

atomic<int> a_cnt,b_cnt,c_cnt;

class A {
public:
   gc_ptr<A> next;
   A()         { a_cnt++; /* printf("constructor A %d\n",a_cnt); */ }
   A(const A&) { a_cnt++; /* printf("constructor A %d\n",a_cnt); */ }
   ~A()        { a_cnt--; /* printf("destructor A %d\n" ,a_cnt); */ }
};

class B  {

   gc_ptr<A> a;

public:
   B()         { a=make_gc<A>(); b_cnt++; /* printf("constructor B %d\n",b_cnt); */ }
   B(const B&) { b_cnt++; /* printf("constructor B %d\n",b_cnt); */ }
   ~B()        { b_cnt--; /* printf("destructor B %d\n" ,b_cnt); */ }
};

class C: public B, public A {
public:
   C()         { c_cnt++; /* printf("constructor C %d\n",c_cnt); */ }
   C(const C&) { c_cnt++; /* printf("constructor C %d\n",c_cnt); */ }
   ~C()        { c_cnt--; /* printf("destructor C %d\n" ,c_cnt); */ }
};

void *test_thread(void *arg)
{
   int i;
   for(i=0;i<CNT;i++) {
      gc_ptr<A> a=make_gc<A>();
      a->next=make_gc<A>();
   }
   return 0;
}


int main()
{

   printf("--- object allocation ---\n");
   {
      gc_ptr<A> a=make_gc<A>();
      assert(a_cnt==1);
   }

   gc_collect();
   assert(a_cnt==0);

   printf("--- array allocation ---\n");
   {
      gc_ptr<A> a=make_gc<A[]>(10);
      assert(a_cnt==10);
   }

   gc_collect();
   assert(a_cnt==0);


   printf("--- shared access ---\n");
   {
      gc_ptr<A> a=make_gc<A>();
      gc_ptr<A> b(a);
      assert(a_cnt==1);
      gc_collect();
      assert(a_cnt==1);
      a=0;
      gc_collect();
      assert(a_cnt==1);
      b=0;
      gc_collect();
      assert(a_cnt==0);
   }
   printf("--- operator= --- \n");
   {
      gc_ptr<A> a,b,c;
      a=make_gc<A>();
      c=b=a;
      a=0;
      b=0;
      assert(a_cnt==1);
      c=0;
      gc_collect();
      assert(a_cnt==0);
   }

   printf("--- cast ---\n");
   {
      gc_ptr<A> a=make_gc<C>();
      assert(c_cnt==1);
      a=0;
      gc_collect();
      assert(c_cnt==0);
   }
   printf("--- cast ---\n");
   {
      gc_ptr<C> a;
      a=make_gc<C[]>(2);
      assert(c_cnt==2);
      a=0;
      gc_collect();
      assert(c_cnt==0);
   }
   printf("--- cast ---\n");
   {
      gc_ptr<C> c;
      c=make_gc<C[]>(2);
      gc_ptr<A> a(c);
      assert(c_cnt==2);
      a=0;
      gc_collect();
      assert(c_cnt==2);
      c=0;
      gc_collect();
      assert(c_cnt==0);
   }
   printf("--- cast ---\n");
   {
      gc_ptr<C> c;
      c=make_gc<C[]>(2);
      gc_ptr<A> a;
      a=c;
      assert(c_cnt==2);
      a=0;
      gc_collect();
      assert(c_cnt==2);
      c=0;
      gc_collect();
      assert(c_cnt==0);
   }

   printf("--- circular ---\n");
   {
      gc_ptr<A> a=make_gc<A>();
      a->next=make_gc<A>();
      a->next->next=a;
      assert(a_cnt==2);
      a=0;
      gc_collect();
      assert(a_cnt==0);
   }

   printf("--- increment decrement \n");
   {
      gc_ptr<char> p=make_gc<char[]>(30);
      strcpy(p.get(),"abcdefghij");

      gc_ptr<char> q;
      for(q=p; *q; ++q) {
         (*q)++;
      }

      assert(!strcmp(p,"bcdefghijk"));
      assert(q!=0);

      for(; q-->p; ) {
         (*q)--;
      }

      assert(!strcmp(p,"abcdefghij"));
      assert(q+1==(char *)p);
      assert(q==p-1);
      assert(p-q==1);
      p=0;
      q=0;
   }

   printf("--- Timing ---\n");
   {
      int i;

      struct timespec start,end;

      clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
      for(i=0;i<CNT;i++) {
         gc_ptr<A> a=make_gc<A>();
      }
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
      end.tv_nsec-=start.tv_nsec;
      if(end.tv_nsec<0) { end.tv_nsec+=1000000000; end.tv_sec--; }
      end.tv_sec-=start.tv_sec;
      printf("%d times create pointer + new: %d.%09d\n",CNT,(int)end.tv_sec,(int)end.tv_nsec);

      clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
      gc_ptr<A> a;
      for(i=0;i<CNT;i++) {
         a=make_gc<A>();
      }
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
      end.tv_nsec-=start.tv_nsec;
      if(end.tv_nsec<0) { end.tv_nsec+=1000000000; end.tv_sec--; }
      end.tv_sec-=start.tv_sec;
      printf("%d times new: %d.%09d\n",CNT,(int)end.tv_sec,(int)end.tv_nsec);
      printf("gc_alloc_cnt: %d\n",int(a_cnt));

      clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
      gc_ptr<A> b=a;
      for(i=0;i<CNT;i++) {
         a->next=make_gc<A>();
         a=a->next;
      }
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
      end.tv_nsec-=start.tv_nsec;
      if(end.tv_nsec<0) { end.tv_nsec+=1000000000; end.tv_sec--; }
      end.tv_sec-=start.tv_sec;
      printf("%d times new chain: %d.%09d\n",CNT,(int)end.tv_sec,(int)end.tv_nsec);
      printf("gc_alloc_cnt: %d\n",int(a_cnt));

      a=0;
      b=0;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
      gc_collect();
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
      end.tv_nsec-=start.tv_nsec;
      if(end.tv_nsec<0) { end.tv_nsec+=1000000000; end.tv_sec--; }
      end.tv_sec-=start.tv_sec;
      printf("gc_collect %d.%09d\n",(int)end.tv_sec,(int)end.tv_nsec);

      clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
      for(i=0;i<CNT;i++) {
         shared_ptr<A> a(new A);
      }
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
      end.tv_nsec-=start.tv_nsec;
      if(end.tv_nsec<0) { end.tv_nsec+=1000000000; end.tv_sec--; }
      end.tv_sec-=start.tv_sec;
      printf("%d times create shared pointer + new: %d.%09d\n",CNT,(int)end.tv_sec,(int)end.tv_nsec);

      clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
      A *c;
      for(i=0;i<CNT;i++) {
         c=new A;
         delete c;
      }
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
      end.tv_nsec-=start.tv_nsec;
      if(end.tv_nsec<0) { end.tv_nsec+=1000000000; end.tv_sec--; }
      end.tv_sec-=start.tv_sec;
      printf("%d times new + delete %d.%09d\n",CNT,(int)end.tv_sec,(int)end.tv_nsec);
   }

#if defined _REENTRANT && _REENTRANT==1
   printf("------------------ multithreading ------------------\n");

   {
      struct timespec start,end;

      pthread_t t[THREADS];

      clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);

      for(unsigned i=0;i<THREADS;i++) pthread_create(&t[i],NULL,test_thread,NULL);
      for(unsigned i=0;i<THREADS;i++) pthread_join(t[i],NULL);

      clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
      end.tv_nsec-=start.tv_nsec;
      if(end.tv_nsec<0) { end.tv_nsec+=1000000000; end.tv_sec--; }
      end.tv_sec-=start.tv_sec;
      printf("%d times new + delete %d.%09d\n",10*CNT,(int)end.tv_sec,(int)end.tv_nsec);
   }
   gc_collect();
   assert(a_cnt==0);
#endif

   return 0;
}
