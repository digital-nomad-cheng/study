#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

vector<int> parseLine(string line)
{
    istringstream ss(line);
    int n;
    char c;
    vector<int> row;
    while(ss >> n >> c  && c == ',') {
        row.push_back(n);
    }
    return row;
}

vector<vector<int>> readBoardFile(string path)
{
    ifstream fs(path);
    vector<vector<int>> board;
    string line;
    if (fs) {
        while(getline(fs, line)) {
            vector<int> int_line = parseLine(line);
            board.push_back(int_line);
        }
    } else {
        cout << "Failed to open file" << "\n";
    }
    return board;
}

void PrintBoard(const vector<vector<int>> board) {
  for (int i = 0; i < board.size(); i++) {
    for (int j = 0; j < board[i].size(); j++) {
      cout << board[i][j];
    }
    cout << "\n";
  }
}

int main() {
    vector<vector<int>> board = readBoardFile("1.board");
    PrintBoard(board);
}
