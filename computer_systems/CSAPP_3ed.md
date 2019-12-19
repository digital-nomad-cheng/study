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





