#include <iostream>

class Date
{
public:
    int day;
    int month;
    int year;

    void setDate(int day, int month, int year);
};

void Date::setDate(int day, int month, int year)
{
    int day_number[]{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

    if ((year % 4 == 0) && (year % 100 != 0)) {
        day_number[1] = 29;
    }

    if (year < 1 || day < 1 || month < 1 || month > 12 || 
        day > day_number[month-1])
        throw std::domain_error("Invalid date!");

    Date::day = day;
    Date::month = month;
    Date::year = year;

}

int main()
{
    Date date;
    date.setDate(28, 2, 2020);
    std::cout << date.day << "/" << date.month << "/" << date.year << "\n";
}
