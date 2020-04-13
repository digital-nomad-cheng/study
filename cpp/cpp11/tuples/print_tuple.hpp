#include <tuple>
#include <iostream>

/* This code make heavy use of template metaprogramming to recutsively iterate at comple
   time over the elements of a tuple. Each call of PRINT_TUPLE<>::print() prints one element 
   and calls the same function for the next element. A partial specialization, where the 
   current index IDX and the number of elements in the tuple MAX are equal, ends this recursion.
 */
// helper: print element with index IDX of tuple with MAX elements
template <int IDX, int MAX, typename... Args>
struct PRINT_TUPLE {
	static void print (std::ostream& strm, const std::tuple<Args...>& t) {
		strm << std::get<IDX>(t) << (IDX+1==MAX ? "" : ",");
		PRINT_TUPLE<IDX+1, MAX, Args...>::print(strm, t);
	}
};

// partial specialization to end recursion
template <int MAX, typename... Args>
struct PRINT_TUPLE<MAX, MAX, Args...> {
	static void print(std::ostream& strm, const std::tuple<Args...>&t) {
	}
};

// output operator for tuples
template <typename... Args>
std::ostream& operator << (std::ostream& strm, const std::tuple<Args...>&t)
{
	strm << "[";
	PRINT_TUPLE<0, sizeof...(Args), Args...>::print(strm, t);
	return strm << "]";
}
