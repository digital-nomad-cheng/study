#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>


using namespace std;

enum class State {kEmpty, kObstacle, kClosed, kPath, kStart, kFinish};
const int delta[4][2] {{-1, 0}, {0, -1}, {1, 0}, {0, 1}};

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
        case State::kObstacle:
            return "⛰   ";
        case State::kPath:
            return "🚗 ";
        case State::kStart:
            return "🚦 ";
        case State::kFinish:
            return "🏁 ";
        default: 
            return "0  ";
    }
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


void cellSort(vector<vector<int>> *v) {
    sort(v->begin(), v->end(), compare);
}

int heuristic(int x1, int y1, int x2, int y2)
{
    return abs(x2-x1) + abs(y2-y1);
}

void addToOpen(int x, int y, int g, int h, vector<vector<int>> &open_nodes, 
               vector<vector<State>> &grid)
{
    vector<int> node {x, y, g, h};
    open_nodes.push_back(node);
    grid[x][y] = State::kClosed;
}

bool  checkValidCell(int x, int y, vector<vector<State>> &grid)
{
    int rows = grid.size();
    int cols = grid[0].size();

    if (x >= rows or y >= cols) {
        // outside of the grid
        return false;
    }

    if (grid[x][y] == State::kEmpty) {
        return true;
    }
    return false;
}

void expandNeighbors(vector<int> curnode, vector<vector<int>> &open_nodes,
                     vector<vector<State>> &grid, int goal[2])
{
    
    int x = curnode[0];
    int y = curnode[1];
    
    // loop through neighbors
    for (int i = 0; i < 4; i++) {
        int dx = delta[i][0];
        int dy = delta[i][1];
        
        int new_x = x + dx;
        int new_y = y + dy;
        if (checkValidCell(new_x, new_y, grid)) {
            int new_g = curnode[2] + 1;
            int new_f = heuristic(new_x, new_y, goal[0], goal[2]);
            // open_nodes.push_back(vector<int> {new_x, new_y, new_g, new_f});
            addToOpen(new_x, new_y, new_g, new_f, open_nodes, grid);
        }
    }
}

vector<vector<State>> search(vector<vector<State>> grid, int start[2], int goal[2])
{
    cout << "Fine";
    vector<vector<int>> open{};
     
    int x = start[0];
    int y = start[1];
    int g = 0;
    int h = heuristic(x, y, goal[0], goal[1]);
    addToOpen(x, y, g, h, open, grid);
    
    while (!open.empty()) {
        cellSort(&open);
        vector<int> curnode = open.back();
        open.pop_back();
        x = curnode[0];
        y = curnode[1];
        grid[x][y] = State::kPath;
        if (x == goal[0] && y == goal[1]) {
            return grid;
        } else {
           // Todo: expand search to current node's neighbors
           expandNeighbors(curnode, open, grid, goal); 
        }
    } 
    
    cout << "No path found!" << "\n"; 
    return std::vector<vector<State>>{};;
}

int main()
{

    int start[2] = {0, 0};
    int goal[2] = {1, 1};
    vector<vector<State>> board = readBoardFile("1.board");
    printBoard(board);
    // vector<vector<State>> solution = search(board, start, goal);
    // printBoard(solution); 
    // return 0;


}
