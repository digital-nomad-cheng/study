gcc -c -o main.o -std=c89 -Wextra main.c
g++ -c -o cpp.o -std=c++98 cpp.cpp
g++ -o main.out main.o cpp.o
./main.out