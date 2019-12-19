##  

## minimal

```c
int main()
{
	int i = 1337;
	return 0;
}
```

1. compile with gdb: `gcc -g minimal.c -o minimal`

2. setup breakpoint: `breakpoint set --name main`

3. start program: `run`

4. execute current line: `next`

5. Examine memory: `x`

   ```x/4xb &i```

6. `ptype` get type

7. set value

   ```shell
   set var u = 0x12345678
   x/4xb &i
   ```

## arrays

```c
int main()
{
    int a[] = {1,2,3};
    return 0;
}
```

1. `gcc -g arrays.c -o arrays`
2. when array name is passed to sizeof and when the array name is passed to the & operator array name doesn't decay to a pointer to the array's first element.
3. pointer a actually decays to is &a[0]

## assembly

```c
// simple.c
int main() 
{
	int a = 5;
	int b = a+6;
	return 0;
}
```

1. compile: `CFLAGS="-g -O0" make simple

2. disassemble: assembly code

   + mnemonic: human readable name for the instruction
   + source
   + destination

   source and destination are operands and can be immediate values, registers and memory addresses or labels. constants prefixed by a $ and register names are prefixed by a %

3. register

   + special: %eax, %ecx
   + general: %rbp, %rsp

   %rbp: base pointer, points to the base of the current stack frame

   %rsp: stack pointer points to the top of the current stack frame

4. regitest backward compatibility

   ```
   |__64__|__56__|__48__|__40__|__32__|__24__|__16__|__8___|
   |__________________________RAX__________________________|
   |xxxxxxxxxxxxxxxxxxxxxxxxxxx|____________EAX____________|
   |xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|_____AX______|
   |xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|__AH__|__AL__|
   ```

5. translation

   ```c
   Dump of assembler code for function main:
   0x0000000100000f50 <main+0>:    push   %rbp
   0x0000000100000f51 <main+1>:    mov    %rsp,%rbp
   0x0000000100000f54 <main+4>:    mov    $0x0,%eax // return value stored at %eax
   0x0000000100000f59 <main+9>:    movl   $0x0,-0x4(%rbp)
   0x0000000100000f60 <main+16>:   movl   $0x5,-0x8(%rbp)
   0x0000000100000f67 <main+23>:   mov    -0x8(%rbp),%ecx
   0x0000000100000f6a <main+26>:   add    $0x6,%ecx
   0x0000000100000f70 <main+32>:   mov    %ecx,-0xc(%rbp)
   0x0000000100000f73 <main+35>:   pop    %rbp // pop old base pointer off the stack
   0x0000000100000f74 <main+36>:   retq   // jumps back to return address
   End of assembler dump.
   ```

   

   

## static local variables

```c
/* static.c */
#include <stdio.h>
int natural_generator()
{
        int a = 1;
        static int b = -1;
        b += 1;
        return a + b;
}

int main()
{
        printf("%d\n", natural_generator());
        printf("%d\n", natural_generator());
        printf("%d\n", natural_generator());

        return 0;
}
```

1. build: `CGLAGS="-g -O0" make static`

2. disassemble

   ```c
   Dump of assembler code for function natural_generator:
   push   %rbp
   mov    %rsp,%rbp
   movl   $0x1,-0x4(%rbp)
   mov    0x177(%rip),%eax        # 0x100001018 <natural_generator.b>
   add    $0x1,%eax
   mov    %eax,0x16c(%rip)        # 0x100001018 <natural_generator.b>
   mov    -0x4(%rbp),%eax
   add    0x163(%rip),%eax        # 0x100001018 <natural_generator.b>
   pop    %rbp
   retq   
   End of assembler dump.
   ```







## Reference

1. lldb: https://lldb.llvm.org/use/tutorial.html

2. https://www.recurse.com/blog/7-understanding-c-by-learning-assembly#footnote_p7f1

3. https://www.recurse.com/blog/5-learning-c-with-gdb

   