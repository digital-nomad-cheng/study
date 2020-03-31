#New Features

1. Spaces in Template Expressions

   ```c++
   vector<list<int> >; // OK in each C++ version
   vector<list<int>>; // OK since C++11
   ```

2. nullptr and std::nullptr_t

   C++11 lets you use nullptr instead of 0 or NULL to specify that a pointer to no value

   ```c++
   void f(int);
   void f(void*);
   f(0);
   f(NULL);
   f(nullptr);
   // calls f(int)
   // calls f(int) if NULL is 0, ambiguous otherwise // calls f(void*)
   ```

3. Automatic Type Deduction with auto

4. Uniform initialization and iniitialier lists

   + force `value initialization` , local variables of fundamental data types which usually have an undefined initial value, are initialized by zero(or nullptr)
   + Check narrowing initializations
   + Initializer lists for user-defined types: std::initializer_list<>
   + `explicit` keyword

   ```c++
   int values[] { 1, 2, 3 };
   std::vector<int> v { 2, 3, 5, 7, 11, 13, 17 }; std::vector<std::string> cities {
   "Berlin", "New York", "London", "Braunschweig", "Cairo", "Cologne" };
   std::complex<double> c{4.0,3.0}; // equivalent to c(4.0,3.0)
   
   int i; // i has undefined value
   int j{}; // j is initialized by 0
   int* p; // p has undefined value
   int* q{}; // q is initialized by nullptr
   
   int x1(5.3); // OK, but OUCH: x1 becomes 5
   int x2 = 5.3; // OK, but OUCH: x2 becomes 5 
   int x3{5.0};  // ERROR: narrowing
   int x4 = {5.3}; // ERROR: narrowing
   char c1{7}; // OK: even though 7 is an int, this is not narrowing
   char c2{99999}; // ERROR: narrowing (if 99999 doesn’t fit into a char)
   std::vector<int> v1 {1, 2, 4, 5 }; // OK
   std::vector<int> v2 {1, 2.3, 4, 5.6 }; // ERROR: narrowing doubles to ints
   
   
   void print (std::initializer_list<int> vals) {
   	for (auto p=vals.begin(); p!=vals.end(); ++p) { // process a list of values
     	std::cout << *p << "\n";
   	} 
   } 
   print ({12,3,5,7,11,13,17}); // pass a list of values to print()
   
   ```

5. Ranged based for loops

   ```c++
   for ( decl : coll ) { 
   	statement
   }
   
   template <typename T>
   void printElements (const T& coll)
   {
   	for (const auto& elem : coll) {
   		std::cout << elem << std::endl;
   	} 
   }
   
   for (auto _pos=coll.begin(); _pos != coll.end(); ++_pos ) {
   	const auto& elem = *_pos;
   	std::cout << elem << std::endl;
   }
   
   for (auto _pos=begin(coll), _end=end(coll); _pos!=_end; ++_pos ) {
   	decl = *_pos;
   	statement
   }
   
   ```

6. Move semantics and Rvalue References

    To avoid unnecessary copies and temporaries.

   

   

