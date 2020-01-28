#include <iostream>
#include <string>

// input parameters of constant referenced pointer to char date type
char *addSpaces(char *str)
{
    char *temp = new char(sizeof(str)*2);
    char *start = temp;
    while (*str != '\0') {
        *temp++ = *str++;
        *temp++ = ' ';
    }
    // str = start;
    str = start;
    
    return start;
}

int main()
{
    std::string str = "Hello World";
    const char *ptr = str.c_str();
    std::cout << "string is: " << ptr << std::endl;
    
    char *ptr2 = new char [str.length()+1];
    std::strcpy(ptr2, str.c_str());

    std::cout << "Value of ptr2" << (void*)&ptr2[0] << std::endl;
    
    char *new_ptr = addSpaces(ptr2); 

    std::cout << "Value of ptr2" << (void *)&ptr2[0] << std::endl;
     
    std::cout << "string is: " << new_ptr << std::endl;
    std::cout << "string is: " << ptr << std::endl;
    std::cout << "string is: " << ptr2 << std::endl;
     
    return 0;
}

