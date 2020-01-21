#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

/*
Enums
C++ allows you to define a custom type which has values limited to a specific 
range you list or "enumerate". This custom type is called an "enum".
Suppose you were writing a program that stores information about each user's 
car, including the color. You could define a Color enum in your program, 
with a fixed range of all the acceptable values:
We want to limited the possible colors.
    white
    black
    blue
    red
https://en.cppreference.com/w/cpp/language/enum
scoped enums
    => enum + class/structure + name {items}
unscoped enums (only remove the class/sturcture from scoped enums)
    => enum + name {items}
*/

enum class State {kEmpty, kObstacle};

vector<State> parseLine(string line)
{
    istringstream sline(line);
    int n;
    char c;
    vector<State> row;
    State curstate;
    while (sline >> n >> c && c == ',') {
        if (n==0) {
            curstate = State::kEmpty;
        } else if (n==1) {
            curstate = State::kObstacle;
        }
        row.push_back(curstate);
    }
    return row;
}

vector<vector<State>> readBoardFile(string path) {
  ifstream myfile (path);
  vector<vector<State>> board{};
  if (myfile) {
    string line;
    while (getline(myfile, line)) {
      vector<State> row = parseLine(line);
      board.push_back(row);
    }
  }
  return board;
}

string cellString(State state)
{
    switch (state) {
        case State::kEmpty:
            return "0  ";
            break;
        case State::kObstacle:
            return "⛰   ";
            break;
    }

    /* Note we can also use default here
    switch(state) {
        case State::kObstacle:
            return "⛰   ";
        default:
            return "0  ";
    }
     */
    
}

void printBoard(const vector<vector<State>> board) {
  for (int i = 0; i < board.size(); i++) {
    for (int j = 0; j < board[i].size(); j++) {
      cout << cellString(board[i][j]);
    }
    cout << "\n";
  }
}

int main() {
  vector<vector<State>> board = readBoardFile("1.board");
  printBoard(board);
}

