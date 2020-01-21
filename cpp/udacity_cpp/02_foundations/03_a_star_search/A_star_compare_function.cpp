#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

enum class State {kEmpty, kObstacle, kClosed};

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
        case State::kClosed:
            return "c  ";
            break;
    }

    /* Note we can also use default here
    switch(state) {
        case State::kObstacle:
            return "⛰   ";
        default:
            return "0  ";
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

bool compare(vector<int> first, vector<int> second)
{
    if (first[3] + first[4] > second[3] + second[4]) {
        return true;
    } else {
        return false;
    }
}

int heuristic(int x1, int y1, int x2 , int y2)
{
    return abs(x2-x1) + abs(y2-y1);
}

void addToOpen(int x, int y, int g, int h, vector<vector<int>> open_nodes, 
               vector<vector<State>> grid)
{
    vector<int> node {x, y, g, h};
    open_nodes.push_back(node);
    grid[x][y] = State::kClosed;
}

vector<vector<State>> search(vector<vector<State>> grid, int start[2], 
                             int goal[2])
{
    cout << "No path found!" << endl;
    vector<vector<int>> open;

    int x = start[0];
    int y = start[1];
    int g = 0;
    int h = heuristic(x, y, goal[0], goal[1]);
    addToOpen(x, y, g, h, open, grid);
     
    cout << "Path not found" << endl;

    vector<vector<State>> path {};
    return path;
}

int main()
{
    int start[2] = {0, 0};
    int goal[2] = {4, 5};

    vector<vector<State>> board = readBoardFile("1.board");
    vector<vector<State>> solution = search(board, start, goal);
    printBoard(solution); 
    return 0;
}

