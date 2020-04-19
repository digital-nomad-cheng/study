1. Modern initialization using curly brances
```c++
int i{42};
std::string s{"hello"};
```
Why list iniitialization is better?
[Look at the stackoverflow answer Here](https://stackoverflow.com/questions/18222926/why-is-list-initialization-using-curly-braces-better-than-the-alternatives)
Basically, list initialization is more safe for it doesn't allow narrowing.

2. Structured biindings: initialize multiple entities by the elements of members of an object
```c++
struct MyStruct {
	int i = 0;
	std::string s;
}
MyStruct ms;

auto [u, v] = ms;
```

+ Helpful for functions returning structures or arrays
```c++
MyStruct getStruct() {
     return MyStruct{42, "hello"};
}
auto[id,val] = getStruct(); // id and val name i and s of returned struct
```
+ Make code more readable by binding the value directly to names that convey semantic meaning about their purpose.
```c++
for (const auto& elem : mymap) {
	std::cout << elem.first << ": " << elem.second << '\n';
}

for (const auto& [key,val] : mymap) { 
	std::cout << key << ": " << val << '\n';
}
```
3. Nested namespace

```c++
// c++11
#include <iostream>
namespace X
{
namespace Y
{
namespace Z 
{
	auto msg = "Hello World\n";
}
}
}

// C++17
namespace X::Y::Z
{
       auto msg = "Hello World\n";
}
```
4. Inline Variables: Avoids the need for managing the cubesome extern variables
```c++
#include <iostream>
inline auto msg = "Hello World\n";
int main(void)
{
	std::cout << msg;
}

// instead of
// expose the variabel
extern const char *msg;
// define the variable
const char *msg = "Hello World\n";
```
5. std::string_view: A wrapper around a character array, similar to std::array, helps to make working with basic C Strings safer and easier.
+ basic accessors
+ reduce size
+ find
+ search
```c++
#include <iostream>
#include <string_view>
int main(void)
{
	std::string_view str("Hello World");
	std::cout << str.size() << '\n'; 
	std::cout << str.max_size() << '\n'; 
	std::cout << str.empty() << '\n';

	std::cout << str.front() << '\n'; 
	std::cout << str.back() << '\n'; 
	std::cout << str.at(1) << '\n'; 
	std::cout << str.data() << '\n';
}
```


