#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

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

void printBoard(const vector<vector<State>> board)
{
    for (int i = 0; i < board.size(); i++) {
        for (int j = 0; j < board[i].size(); j++) {
            cout << cellString(board[i][j]);
        }
        cout << "\n";
    }
}

vector<vector<State>> search(int start[2], int goal[2])
{
    cout << "No path found!" << endl;
    vector<vector<State>> path {};
    return path;
}

int main()
{
    int start[2] = {0, 0};
    int goal[2] = {4, 5};

    vector<vector<State>> solution;
    solution = search(start, goal);
    printBoard(solution); 
    return 0;
}

