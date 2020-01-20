#include <iostream>
#include <vector>

using std::cout;
using std::vector;

void printBoard(vector<vector<int>> board)
{
    if (board.empty())
        return; 
    
    int row = board.size();
    int col = board[0].size();

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            cout << board[i][j];
        cout << "\n";
    }
}

int main()
{
    vector<vector<int>> board{{0, 1, 0, 0, 0, 0},
                              {0, 1, 0, 0, 0, 0},
                              {0, 1, 0, 0, 0, 0},
                              {0, 1, 0, 0, 0, 0},
                              {0, 0, 0, 0, 1, 0}};

    printBoard(board);
}

