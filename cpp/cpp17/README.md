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