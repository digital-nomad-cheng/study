#include <iostream>
#include <string>

using namespace std;
class Student
{
public:
    Student(string name, int grade, float GPA)
    {
        if (grade >= 0 && GPA >= 0.0 && GPA <= 4.0) {
            this->name = name;
            this->grade = grade;
            this->GPA = GPA;
        } else {
            throw std::domain_error("value out of range");
        }
    }

    void setName(string name)
    {
        this->name = name;
    }
    void setGrade(int grade)
    {
        if (grade < 0)
            throw std::domain_error("grade must not be negative");
        
        this->grade = grade;      
    }
    void setGPA(float GPA)
    {
        if (GPA < 0.0 || GPA >4.0)
            throw std::domain_error("GPA should be within range [0, 4]");
        this->GPA = GPA;
    }


private:
    string name;
    int grade;
    float GPA;

};

int main()
{
    Student s1("A", 10, 4.0);
    s1.setGPA(5.0);
}
