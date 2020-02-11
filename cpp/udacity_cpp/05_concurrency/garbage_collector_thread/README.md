# CppND-Garbage-Collector
The final project for this Memory Management course is to implement your own
version of a smart pointer. You can think of this as implementing your own
garbage collector, to use a concept from other programming languages. Building
this project will help you understand both how to work with pointers and
references, and also why smart pointers are so important to modern C++
programming. Complete the implementations and verify that your implementation
does not have any memory leaks!

# Building
To build this project, you will need a C++ compiler. The `make` script provided
assumes the GCC compiler, however you may substitute an equivalent C++ compiler
by adjusting the `make` script as needed. Execute the make script, then run the
compiled executable.

If the code fails to compile, the execute won't be created or will remain the
last-compiled version. Adjust your code to resolve compiler errors and try again.

``` shell
$ ./make
$ ./compiled
```

## Project TODO List:
- Complete `Pointer` constructor
- Complete `Pointer` `operator==`
- Complete `Pointer` destructor
- Complete `PtrDetails` class

## Reference
- [Chaper 2 - A Simple Garbage Collector for C++, The Art of C++.](https://www.cmlab.csie.ntu.edu.tw/~chenhsiu/tech/The_Art_of_C++_ch2.pdf)

## Garbage Collector Algorithm 
+ Reference Counter: simple, independent of an object's physical location, add overhead to each pointer operation, but the collection phase is relatively low cost
+ Mark and Sweep: First phase, all objects in the heap are set to unmarked state, all objects directly or indirectly accessible from program variables are marked as `in-use`. Phase two, all of allocated memory is scanned, and all unmarked elements are released.
+ Copy: 

+ Multithread or not?

## Function
+ can only be used when creating local objects
+ must specify size when used with arrays
+ avoid cicular pointer references


## Todo
+ overload new so that it automatically collects garbage when an allocation failure occurs
+ Perform garbage collection on seperate thread

https://meixnergc.sourceforge.io/ : C++11 multi-thread garbage collection with 
mark-swap and reference collection

