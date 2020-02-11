#include <iostream>
#include <new>
#include "gc_pointer_thread.hpp"

class LoadTest
{
	int a, b;
public:
	double n[10000];
	double val;

	LoadTest() { a = b = 0; }
	LoadTest(int x, int y) {
		a = x;
		b = y;
		val = 0.0;
	}
	friend std::ostream &operator<<(std::ostream &strm, LoadTest &obj);
};

std::ostream &operator<<(std::ostream &strm, LoadTest &obj) {
	strm << "(" << obj.a << " " << obj.b << ")";
	return strm;
}

int main()
{
	Pointer<LoadTest> mp;
	int i;
	for (i = 0; i < 2000; i++) {
		try {
			mp = new LoadTest(i, i);
			// if (!(i%100)) {
			mp.showList();
			std::cout << "gc_list contains: " <<  mp.ref_container_size() 
						  << "entries.\n" << std::endl;
			// }
		} catch (std::bad_alloc xa) {
			std::cout << "last object:" << *mp << std::endl;
			std::cout << "Length of gc_list: " << mp.ref_container_size() << std::endl;
		}

	}
	return 0;
} 