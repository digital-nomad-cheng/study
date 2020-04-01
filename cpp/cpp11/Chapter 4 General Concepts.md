#Chapter 4 General Concepts

1. namespace
    - use directly

            std::cout << std::hex << 3.4 << std::endl;

    - using declaration

            using std::cout;
            using std::endl;
            cout << std::hex << 3.4 << endl;

    - using directive: Bad!

            using namespace std;
            cout << hex << 3.4 << endl;

2. header files
    - no extension
    - prefix `c` for C header files
3. Error and Exception Handling

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d3e8b9f8-b7ca-457e-b0e3-c0fa006803f7/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d3e8b9f8-b7ca-457e-b0e3-c0fa006803f7/Untitled.png)

4. Callable objects
5. Concurrency and Multithreading