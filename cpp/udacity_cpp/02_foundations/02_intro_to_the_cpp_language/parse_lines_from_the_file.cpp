#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

vector<int> ParseLine(string line)
{
    vector<int> rst;
    istringstream my_reader(line);
    int n;
    char c;
    while (my_reader >> n >> c) {
        rst.push_back(n);
    }
    return rst;
}

void readBoardFile(string path)
{
    ifstream fs(path);
    if (fs) {
        string line;
        while(getline(fs, line)) {
            cout << line << "\n";
        }
    } else {
        cout << "Failed to read file from path" << endl;
    }
}

#include "test.cpp"
int main() 
{
    TestParseLine();
}
