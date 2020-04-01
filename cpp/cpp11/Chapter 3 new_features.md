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

7. New String Literals  
   ```c++
   // ordinary string
   "\\\\n"
   // string literal
   R"(\\n)"

   ```

8. Keyword noexcept: used to specify that a function cannot throw or is not prepared to throw.
   ```C++
   void foo() noexcept;
   ```
   If an exception is not handled locally inside foo() — thus, if foo() throws — the program is terminated, calling std::terminate(), which by default calls std::abort()

9. constexpr: used for compile time evaluation
   ```c++
   constexpr int square (int x) 
   {
      return x * x; 
   }
   float a[square(9)]; // OK since C++11: a has 81 elements
   ```
   + fixs a problem C++98 had when using numeric limits, `std::numeric_limits<short>::max()` cannot be used as an integral constant.

10. variadic template: accept a variable number of template arguments
   ```c++
   void print ()
   {
   }
   template <typename T, typename... Types>
   void print (const T& firstArg, const Types&... args) {
   std::cout << firstArg << std::endl; // print first argument
   print(args...); // call print() for remaining arguments }
   ```

11. Alias Templates with using keyword
   ```c++
   template <typename T>
   using Vec = std::vector<T,MyAlloc<T>>;
   Vec<int> coll;

   // is equivalent to
   std::vector<int,MyAlloc<int>> coll;
   ```

12. Lambdas

13. Keyword decltype using decltype
   - declare return types
   - metaprogramming
   - pass the type of a lambda
   ```c++
   std::map<std::string, float> col;
   decltype(coll)::value_type elem;
   ```

14. New Function Declaration Syntax
   Declare the return type of a function behind the parameter list
   ```c++
   template <typename T1, typename T2>
   auto add(T1 x, T2 y) -> decltype(x+y)
   ```

15. New Fundamental Data types
   - char16_t
   - char32_t
   - long long
   - unsigned long long
   - std::nullptr_t

16. Scoped Enumerations
   - Implicit conversions to and from int are not possible.
   - Values like mr are not part of the scope where the enumeration is declared. You have to use Salutation::mr instead.
   - You can explicitly define the underlying type (char here) and have a guaranteed size (if you skip “: char” here, int is the default).
   - Forward declarations of the enumeration type are possible, which eliminates the need to recompile compilation units for new enumerations values if only the type is used.
   ```c++
   enum class Salutation : char { mr, ms, co, none };
   ```
