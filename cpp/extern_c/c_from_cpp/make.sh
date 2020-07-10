g++ -c -o main.o -std=c++11 main.cpp
gcc -c -o c.o -std=c99 c.c
g++ -o main.out main.o c.o
