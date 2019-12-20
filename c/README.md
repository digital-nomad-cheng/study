## Chapter 05 Points and Arrays

1. `p = &c`

   `&` operator only applies to objects in memory: variables and array elements. It cannot be applied to expressions, constants, or register variables

2. indirection or deferencing operator: `*`

3. A pointer to void is used to hold any type of pointer but cannot be dereferenced itself

4. priority

   ```c
   // add 1 to what ip points to
   y = *ip + 1
   *ip += 1
   (*ip)++
   
   // increase ip instead of what it points to
   *ip++
   ```

5. Pointer arguments enable a function to access and change objects in the function that called it.