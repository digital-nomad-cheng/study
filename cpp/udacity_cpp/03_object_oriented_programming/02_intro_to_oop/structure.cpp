#include <iostream>

// cannnot add logic to structure in c

struct Date
{
    int day;
    int month;
    int year;
/bin/bash: :q: command not found

int main()
{
    Date date;
    date.day = 1;
    date.month = 10;
    date.year = 2019;
    std::cout << date.day << "/" << date.month << "/" << date.year << "\n";
}

