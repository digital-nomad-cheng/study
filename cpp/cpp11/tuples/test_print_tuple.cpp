#include <tuple>
#include <iostream>
#include <string>

#include "print_tuple.hpp"
int main() {
	using namespace std;
	tuple <int, float, string> t(77, 1.1, "more light");
	cout << "io: " << t << endl;
}