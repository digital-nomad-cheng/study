#include <stdio.h>

void echo()
{
    char buf[4];
    gets(buf);
    puts(buf);
}

void call_echo()
{
    echo();
}

int main()
{
    echo();
}
