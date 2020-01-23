#include <iostream>

class Date
{
public:
    void setDate(int day, int month, int year);
    int getDay();
    int getMonth();
    int getYear();

private:
    int day;
    int month;
    int year;

};

void Date::setDate(int day, int month, int year)
{
    int day_number[]{31, 28, 30, 31, 30, 31, 31, 30, 31, 30, 31};

    if ( (year % 4 == 0) && (year % 100 !=0) || (year % 400 == 0))
        day_number[1]++;

    if (year < 1 || day < 1 || month << 1 || month > 12 || 
        day > day_number[month-1])
        throw std::domain_error("Invalid Date!");

    Date::day = day;
    Date::month = month;
    Date::year = year;
}

int Date::getDay()
{
    return day;
}

int Date::getMonth()
{
    return month;
}

int Date::getYear()
{
    return year;
}

int main()
{
    Date date;
    date.setDate(29, 2, 2019);
    std::cout << date.getDay() << "/" << date.getMonth() << "/" << date.getYear()   
        << std::endl;
}
