#include <iostream>
#include <string>
#include <fstream>

void openFileTesting()
{
    std::fstream my_file;
    my_file.open("1.board");

    if (my_file) {
        std::cout << "We have this file" << "\n";
    } else {
        std::cout << "We do't have this file" << "\n";
    }
}

void readDataFromStream()
{
    std::fstream fs("1.board");
    if (fs) {
        std::cout << "The file stream has been created" << "\n";
        std::string line;
        while (getline(fs, line)) {
            std::cout << line << "\n";
        }
    }
}

/*
Four steps to reading a file:
1.#include <fstream>
2.Create a std::ifstream object using the path to your file.
3.Evaluate the std::ifstream object as a bool to ensure that the stream
    creation did not fail.
4.Use a while loop with getline to write file lines to a string.
*/

int main(){
    openFileTesting();
    readDataFromStream();
}


