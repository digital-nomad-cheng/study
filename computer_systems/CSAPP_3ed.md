## Chap02
2.1

+ A: 0x39A7F8 &rarr; 0011 1001 1010 0111 1111 1000
+ B: 1100 1001 0111 1011 &rarr; 0xC97D
+ C: 0xD5E4C &rarr;1011 0101 1110 0100 1100 
+ D: 10 0110 1110 0111 1011 0101&rarr; 0x26E7B5

2.2 

| n    | 2^n(base 10) | 2^n(base 16) |
| ---- | ------------ | ------------ |
| 9    | 512          | 0x200        |
| 19   | 1024*512     | 0x80000      |
| 15   | 16384        | 0x8000       |
| 16   | 65536        | 0x10000      |
| 17   | 131072       | 0x20000      |
| 5    | 32           | 0x20         |
| 7    | 128          | 0x80         |

2.3

| base 10 |  Base 2   | base 16 |
| :-----: | :-------: | :-----: |
|    0    | 0000 0000 |  0x00   |
|   167   | 1010 0111 |  0xA7   |
|   188   | 1011 1100 |  0xBC   |
|   55    | 0011 0111 |  0x37   |
|   136   | 1000 1000 |  0x88   |
|   243   | 1111 0011 |  0xF3   |
|   82    | 0101 0010 |  0x52   |
|   172   | 1010 1100 |  0xAC   |
|   231   | 1110 0111 |  0xE7   |

2.4 

A. 0x503C + 0x8 = 0x5044

B. 0x503C - 0x40 = 0x4FFC

C. 0x503C + 64 = 0x5A0

D. 0x50EA - 0x503C = 0xAE

2.5 

A. little end: 87 big end: 21

B. little end:87 65 big end: 21 43

C: little end: 87 65 43 big end: 21 43 65

2.6

A: 

0x00359141 &rarr; 0011 0101 1001 0001 0100 0001

0x4A564504 &rarr; 0100 1010 0101 0110 0100 0101 0000 0100 

B: 										

​			      001**101011001000101000001**

​	 010010100**101011001000101000001**00 

21 bits are equal

C: 

​	begin and end don't match

2.7 

61 62 63 64 65 66 00

2.8

| Operation |   Result    |
| :-------: | :---------: |
|     a     | [0110 1001] |
|     b     | [0101 0101] |
|    ~a     | [1001 0110] |
|    ~b     | [1010 1010] |
|    a&b    | [0100 0001] |
|   a\|b    | [0111 1101] |
|    a^b    | [0011 1100] |

2.9

A.

Black &rarr; White

Blue &rarr; Yellow

Green &rarr; Magenta

Cyan &rarr; Red

B. 

Blue | Green = Cyan

Yellow & Cyan = Green

Red ^ Magenta = Blue

2.10

This question is interesting

|  Step   |  *x  |  *y  |
| :-----: | :--: | :--: |
| initial |  a   |  b   |
|  step1  |  a   | a^b  |
|  step2  |  b   | a^b  |
|  step3  |  b   |  a   |

2.11 

A. first = last = k

B. a^a = 0

C. first < last

2.12 

A. x&0xFF

B. ~(x & (1-0xFF)) | (x&0xFF)

C. x | 0xFF

2.13

````c
/* Declarations of functions implementing operations bis and bic */
int bis(int x, int m);
int bic(int x, int m);

// x 0100
// y 1101
// result 1101
int bool_or(int x, int y) {
  // result 1101 
  int result = bis(x, y);
  result = bis(result, x);
  // result 1101
  return result;
}


// result 1001
int bool_xor(int x, int y) {
  // result 1101
  int result = bool_or(x, y);
  // and 0100
  int and = bic(x, y);
  // result 1001
  result = bis(result, and)
 	return x;
}
````

2.14

0x66 &rarr; 0110 0110

0x39 &rarr; 0011 1001

~x = 1001 1001

~y = 1100 0110

| Expression | Value | Expression | Value |
| :--------: | :---: | :--------: | :---: |
|    x&y     | 0x20  |    x&&y    | 0x01  |
|    x\|y    | 0x7F  |   x\|\|y   | 0x01  |
|  ~x \| ~y  | 0xDF  | ~x \|\| ~y | 0x01  |
|    x&!y    | 0x00  |   x&&~y    | 0x01  |

2.15

!(x^y)

2.16

| Hex  |  Binary   |   a<<2    | a<<2 | a>>3, logic | a>>3, logic | a>>3, arithmetic | a>>3, arithmetic |
| :--: | :-------: | :-------: | :--: | :---------: | :---------: | :--------------: | :--------------: |
| 0xD4 | 1101 0100 | 0101 0000 | 0x50 |  0001 1010  |    0x1A     |    1111 1010     |       0xFA       |
| 0x64 | 0110 0100 | 1001 0000 | 0x90 |  0000 1100  |    0x0C     |    0000 1100     |       0x0C       |
| 0x72 | 0111 0010 | 1100 1000 | 0xC8 |  0001 1100  |    0x1C     |    0001 1100     |       0x1C       |
| 0x44 | 0100 0100 | 0001 000  | 0x10 |  0000 1000  |    0x08     |    0000 1000     |       0x08       |

2.17

| Hex  | Binary | Unsigned | Complement |
| :--: | :----: | :------: | :--------: |
| 0xE  |  1110  |    14    |     -2     |
| 0x0  |  0000  |    0     |     0      |
| 0x5  |  0101  |    5     |     5      |
| 0x8  |  1000  |    8     |     -8     |
| 0xD  |  1101  |    13    |     -3     |
| 0xF  |  1111  |    15    |     -1     |

2.18

|  Hex  | Decimal |
| :---: | :-----: |
| 0x2e0 |   736   |
| -0x58 |   -88   |
| 0x28  |   40    |
| -0x30 |   -48   |
| 0x78  |   120   |
| 0x88  |   136   |
| 0x1f8 |   504   |
| 0xc0  |   192   |
| -0x48 |   -72   |

2.19

|  x   | T2U_4(x) |
| :--: | :------: |
|  -8  |    8     |
|  -3  |    13    |
|  -2  |    14    |
|  -1  |    15    |
|  0   |    0     |
|  5   |    5     |

2.20

2.21

| Expression | Type | Evaluation |
| :--------: | :--: | :--------: |
|            |      |            |
|            |      |            |
|            |      |            |
|            |      |            |
|            |      |            |

2.22 

Simple

2.23

A.

|      w      |   fun1(w)   |   fun2(w)   |
| :---------: | :---------: | :---------: |
| 0x0000 0076 | 0x0000 0076 | 0x0000 0076 |
| 0x8765 4321 | 0x0000 0021 | 0x0000 0021 |
| 0x0000 00C9 | 0x0000 00C9 | 0xFFFF FFC9 |
| 0xEDCB A987 | 0x0000 0087 | 0xFFFF FF87 |

B. func1 extracts low-order 8 bits values of the argument, the range of the result should be 0 ~ 255

```c++
func2 extracts low-order 8 bits values of the argument too, if alse performs signed extension, the result will be a number between -128 and 127
```

2.27

```c
int uadd_ok(unsigned x, unsigned y) {
  unsigned z = x + y;
  if (z>x || z >y) {
    return -1;
  }
  else {
    return 1;
  }
}
```

2.30

```c
int tadd_ok(int x, int y) {
	int z = x +y;
  if (x < 0 && y < 0 && z <= 0) {
    return -1;
  }
  else if (x > 0 && y > 0 && z>=0) {
    return -1;
  }
  return 1;
}
```

2.48 

simple

2.49

$2^{n+1}+1$

$2^{24}+1$

2.50

A. 10.010 &rarr; 10.000

B. 10.011 &rarr; 10.100

C. 10.110 &rarr; 11.000

D. 11.001 &rarr; 11.000

2.51

A. 0.0001 1001 1001 1001 1001 1010

2.52

|   bits   | value |   bits   | Value |
| :------: | :---: | :------: | :---: |
| 011 0000 |   1   | 0111 000 |       |
| 101 1110 | 15/2  |          |       |
| 010 1001 | 25/32 |          |       |
| 110 1111 | 31/2  |          |       |
| 000 0001 | 1/64  |          |       |

## Chap03

3.1 

|     Operand     | Value |
| :-------------: | :---: |
|      %rax       | 0x100 |
|      0x104      | 0xAB  |
|     $0x108      | 0x108 |
|     (%rax)      | 0xFF  |
|     4(%rax)     | 0xAB  |
|  9(%rax, %rdx)  | 0x11  |
| 260(%rcx, %rdx) | 0x13  |
| 0xFC(, %rcx, 4) | 0xFF  |
| (%rax, %rdx, 4) | 0x11  |

3.2

movl

movw

movb

movb

movq

movw

3.3 

+ don't know
+ movq
+ cannot have both source and destination as meomory reference
+ No register as sl
+ cannot have immediate number as destination
+ destination register wrong size
+ movb should be movl

3.4 

+ ````
  movq (%rdi), %rax
  movq %rax, (%rsi)
  ````

+ ```
  movsbl (%rdi) %eax
  movl %eax, (%rsi)
  ```

+ ```
  movsbl (%rdi), %al
  movsbl %al, (%rsi)
  ```

+ ```
  movb (%rdi), %al
  mozbw %al, (%rsi)
  ```

+ ```
  movl (%rdi), %eax
  movb %al, (%rsi)
  ```

  
  

3.5

```c
void decode1(long *xp, long *yp, long *zp) {
  long temp1 = *yp;
  *yp = *xp;
  *xp = *zp;
  *zp = temp1;
}
```

3.6

|         expression          | Result |
| :-------------------------: | :----: |
|     leap 6 (%rax), %rdx     | x + 6  |
|   leap (%rax, %rcx), %rdx   |  x+y   |
| leap (%rax, %rcx, 4), %rdx  |  x+4y  |
| leap 7(%rax, %rax, 8), %rdx |  9x+7  |
|  leap 0xA(, %rcx, 4), %rdx  | 4y+10  |
| Leap 9(%rax, %rcx, 2), %rdx | x+2y+9 |

3.7

`long t = 5x + 2y + 8z`

3.8

|        instruction        | destination | value |
| :-----------------------: | :---------: | :---: |
|     add %rcx, (%rax)      |    0x100    | 0x100 |
|    subq %rdx, 8(%rax)     |    0x108    | 0xA8  |
| imulq $16 (%rax, %rdx, 8) |    0x118    | 0x110 |
|      incq 16 (%rax)       |    0x110    | 0x14  |
|         decq %rcx         |    %rcx     |  0x0  |
|      subq %rdx, %rax      |    %rax     | 0xFD  |

3.9

```
salq 4, %rax

sarq %ecx, %rax 
```

3.10

````
long arith2(long x, long y, long z)
{
  long t1 = x | y;
  long t2 = t1 >> 3;
  long t3 = ~t2;
  lont t4 = z - t3;
}
````

3.11

A. set %rdx to zero

B. Move $0 %rdx

C. 

3.12

```c++
movq %rdx %r8
movq %rdi %rax;
movq $0 %rdx // or movel $0 %edx
divq %rsi
movq %rax (%r8) 
movq %rdx (%rcx)
ret
```

3.13

A. int, a < b

B. short >=

C.unsigned char: <=

D. long, unsigned long, !=

3.14

A. Long

B. short, unsigned short

C. unsigned char

D. int, unsigned int

3.15

A. 0x02 + 0x400fc = 0x400f4

B. 0xf4 + 0x400431 = 0x40025

C. Pop = 0x400545

​	Ja: 0x400543

D. 0x400560

3.16

```
void cond(long a, long *p)
{
	if (!p):
		goto L1;
	if (a>=*p):
		goto L1;
	movq %rdi, (%rsi)
	L1:
		rep, ret
}
```

3.17

```c++
/*
 t = test-expr;
 if (t)
 	goto true;
 else-statement
 goto done;
 true:
 	then-statement
 done:
 */

long asbdiff_se(long x, long y)
{
  long result;
  if (x<y):
  	goto x_lt_y;
 	ge_cnt++;
  result = x - y;
  return result;
 x_lt_y:
  lt_cnnt++;
  result = y -x;
  return result
}
```

3.18:

```c++
/*
short test(short x, short y, short z)
x in %rdi, y in %rsi, z in %rdx
test:
leaq (%rdx,%rsi), %rax 
subq %rdi, %rax
cmpq $5, %rdx
jle .L2
cmpq $2, %rsi
jle .L3
movq %rdi, %rax
idivq %rdx, %rax
ret
.L3:
movq %rdi, %rax 
idivq %rsi, %rax 
ret
.L2:
cmpq $3, %rdx 
jge .L4
movq %rdx, %rax 
idivq %rsi, %rax
.L4:
rep; ret
*/
long test(long x, long y, long z)
{
	long val = x + y - z;
  if (z > 5) {
    if (y > 2) {
      val = x / z;
    } else {
      val = x / y;
    }
  } else if (z < 3) {
    val = z / y;
  }
  return val;
}

```

3.19

A. 30

B. 46

3.20

/

````c++
x  >=0 ? x >> 3: (x+7) >> 3;
````

3.21 

````c++
long test(long x, long y)
{
	long val = 8*x;
  if (y > 0) {
  	if ( x > = y) {
      val = x & y;
    } else {
      val = y - x;
    }
  } else if (y + 2 <=0) {
    val = x + y;
  }
  return val;
}
````

3.22 

3.23

```
x: %rax
y: %rcx
z: %rdx

leap 1(%rcx, %rax), %rax
```

3.24

```c++
long loop_while(long a, long b)
{
  long result = 1;
    while (b < a) {
      result = (a+b)*result;
      a = a + 1;
    }
  return result
}
```

3.25

```c
long loop_while2(long a, long b) 
{
	long result = b;
	while( b ) {
		result = a*result;
		b = b - a;
	}
	return result;
}
```

3.26

A. jump to middle

B.

```c
long fun_a(unsigned long x) {
  long val = 0;
  while (x>0) {
    val = val ^ x;
    x = x >> 1;
  }
  return 1 & val;
}
```

C. 计算参数x奇偶性，奇数个1返回1, 偶数个1返回0

3.27

```c
long fact_for_guarded_do_goto(long n)
{
  long i = 2;
  long result = 1;
  goto test;
loop:
	result *= i;
  i++;
test:
  if (i<=n)
    goto loop
  return result;
}
```

3.28

A.

```c
long fun_b(unsgined long x) {
	long val = 0;
	long i;
	for (i=64; i < !=0; i--) {
		val = (val << 1) || (x & ox01);
		x >> 1;
	}
	return val;
}
```

3.29

3.30

A. 

x = -1

max = -1 + 8 = 7

-1, 0, 1, 2, 4, 5, 7

B. 

.L5: 0, 7

.L7: 2, 4

3.31

```c
void switcher(long a, long b, long c, long *dest)
{
  long val;
  switch(a) {
    case 5:
        c = b ^ 15;
    case 0:
        val = c+112;
      	break;
    case 4:
        val = a;
        break;
    case 2:
    case 7:
      	val = (c+b) << 2;
        break;
    default:
      val = b;
  }
  *dest = val;
}
```

3.32

3.33

```
int procprob(int a, short b, long *u, char *v)
```

3.34

```
9 movq %rdi %rbx: move x -> %rbx

A:called: x+1, x+2, x+3, x+4, x+5
B:stack: x+6, x+7
C: used up alled the callee save registers, 
there are only 6 callee save registers
```

3.35

```
A: x
B: 
long rfun(unsigned long x)
{
	if (x==0) 
		return 0;
	unsigned long nx = x >> 2;
	long rv = rfun(nx);
	return rv + x;
}
```

3.36

```
S | 2 bytes | 14 bytes | xs | xs+2*i|
T | 8 bytes | 24 bytes | xt | xs+8*i|
U | 8 bytes | 48 bytes | xu | xu+8*i|
V | 4 bytes | 32 bytes | xv | xv+4*i|
W | 8 bytes | 32 bytes | xw | xw+8*i|
```

3.37

```
	s+1   | short * | xs+2      | leaq 2(%rdx) %rax         |
	s[3]  | short   | M[xs+2*3] | movew 6(%rdx)  %ax
	&S[i] | short * | xs+2*i    | leaq (%rdx, %rcx, 2) %rax |
S[4*i+1 | short   | M[xs+8*i+2]| movew 2(%rdx, %rcx, 2)  %ax|
	S+i-5 | short * | xs+2i-10  | leaq -10(%rdx, %rcx, 2) %rax |
```

3.38

```
%rdi = i, %rsi = j
%rdx = 8i
%rdx = 8i - i = 7i
%rdx = j + 7i
%rax = j + 4j = 5j
%rdi = i + 5j

%rdx = 8(i+5j)

M = 5, N = 7
```

3.39

```
%rdi+16*4*i = xA + 64i
%rsi+4*%rcx = xB + 4k
xB + 4k + 16*16*4 = xB + 4k + 1024
```

3.40

3.41

```
A.
p: 0 bytes
s.x: 8 bytes
s.y: 12 bytes
next: 20

B.
24 bytes
C.

void sp_init(struct prob *sp)
{
    sp->s.x = sp->s.y;
    sp->p = &(sp->s.y);
    sp->next = sp; 
}
```

3.42

```
long fun(struct ELE *ptr)
{
    long val = 0;
    while (ptr!=NULL) {
        val += ptr->v;
        ptr = ptr->p;
    }
    return val;
}
```

3.43

3.44

```
整体对齐按照最大的那个元素决定
A.
struct P1(int i; char c; int j; char d;};
0, 4, 8, 12
16 bytes in total
4 bytes alignment
B. struct P2(int i; char c; char d; long j;};
0, 4, 5, 8
16 bytes in total
8 bytes alignment
```

3.45

```
A.
a: 0
b: 8
c: 16
d: 24
e: 32
f: 36
g: 40
h: 48

B. 
56 bytes in total

C.
larger elements first then smaller elements
```

3.46

A.

```c++
line0 00 00 00 00 00 40 00 76
line1 01 23 45 67 89 AB CD EF
line2
line3 											 buf = %rsp
line4
```

B.

```c++
line0 00 00 00 00 00 40 00 34 return address
line1 33 32 31 30 39 38 37 36
line2 35 34 33 32 32 30 39 38
line3 37 36 35 34 33 32 31 30	buf = %rsp
line4
```

C.

return to address: 0x400034

D.

register %rbx is overwritten

3.47

A. 32 bytes = 2^13 addresses

B. 2^6

3.48

```
unprotected v: %rsp 24 buf: %rsp 
protected canary %rsp bias 40 v: %rsp bias 8 buff: %rsp bias 16 
In the protected code, local variable v is positioned closer to the top of the stack than buf, and so an overrun of buf will not corrupt the value of v.```
```



## Chap06

1. row choose and col choose use the same pins

   |  size  |  r   |  c   |  bc  |  br  | max(br, bc) |
   | :----: | :--: | :--: | :--: | :--: | :---------: |
   |  16x1  |  4   |  4   |  2   |  2   |      2      |
   |  16x4  |  4   |  4   |  2   |  2   |      2      |
   | 128x8  |  16  |  8   |  4   |  3   |      4      |
   | 512*4  |  32  |  16  |  5   |  4   |      5      |
   | 1024*4 |  32  |  32  |  5   |  5   |      5      |

2. 

   4x10000x400x512 = 8 192 000 000 = 8.192 GB

3.  

   $T_{total} = T_{seek} + T_{rotate} + T_{transfer} = 8 + 1/2*(60/15000)*1000+(60/15000)*(1/500)*1000 = 10ms$

4. T_maxrotate = 6ms

   5 + 3 + 2*6 = 20ms

5. Pass

6. Pass

7.  

   ```c
   int sumarray3d(int a[N][N][N])
   {
     int i, j, k, sum = 0;
     for (i = 0; i < N; i++) {
       for (j = 0; j < N; j++) {
         for (k = 0; k < N; k++) {
           sum += a[i][j][k]
         }
       }
     }
     return sum
   }
   ```

8. Clear1: stride 4*6

   Clear2: stride 4*3

   Clear3: stride Non spatial localiity

   Clear2 > clear1 > clear3

9. 

| cache |  m   |  C   |  B   |  E   |  S   |  t   |  s   |  b   |
| :---: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|   1   |  32  | 1024 |  4   |  1   | 256  |  22  |  8   |  2   |
|   2   |  32  | 1024 |  8   |  4   | 128  |  22  |  7   |  3   |
|   3   |  32  | 1024 |  32  |  32  |  32  |  22  |  5   |  5   |

6.12

```
12～5: tag

4~2: set index

1～0: block offset
```

6.13

```
0 1110 0011 0100

CO: 0
CI: 5
CT: 71
YES: 06
```

6.14

```
0 1101 1101 0101
CO: 1
CI: 5
CT: 6E
NO
valid bit is zero
```

6.15

```
1 1111 1110 0100
CO: 0 
CI: 1
CT: FF
NO
miss
```

6.18

A. 16x16x4x2 = 2048

B. 1024

C 50%

6.19

A.2048

B. 1024

C. 50%

D. 25%

6.20

A. 2048

B. 512

C. 25%

D. 25%

