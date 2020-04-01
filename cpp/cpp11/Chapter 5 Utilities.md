#Chapter 5 Utilities

1. pairs and tuples
    - `pair` is defined as struct instead of class so that all members are public
    - operations:

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d191862c-7964-4ccf-997f-75475a31a807/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d191862c-7964-4ccf-997f-75475a31a807/Untitled.png)

    - tuple like interface since C++11

        ```c++
        typedef std::pair<int,float> IntFloatPair; 
        IntFloatPair p(42,3.14);
        
        std::get<0>(p) // yields p.first
        std::get<1>(p) // yields p.second
        std::tuple_size<IntFloatPair>::value // yields 2
        std::tuple_element<0,IntFloatPair>::type // yields int
        ```

