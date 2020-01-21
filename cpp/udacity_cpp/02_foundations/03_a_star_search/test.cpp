#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "A_star_while_loop.cpp"

using namespace std;

void printVector(vector<int> v) {
  cout << "{ ";
  for (auto item : v) {
    cout << item << " ";
  }
  cout << "}" << "\n";
}

void printVectorOfVectors(vector<vector<int>> v) {
  for (auto row : v) {
    cout << "{ ";
    for (auto col : row) {
      cout << col << " ";
    }
    cout << "}" << "\n";
  }
}

void printVectorOfVectors(vector<vector<State>> v) {
  for (auto row : v) {
    cout << "{ ";
    for (auto col : row) {
      cout << cellString(col) << " ";
    }
    cout << "}" << "\n";
  }
}

void testHeuristic() {
  cout << "----------------------------------------------------------" << "\n";
  cout << "Heuristic Function test: ";
  if (heuristic(1, 2, 3, 4) != 4) {
    cout << "failed" << "\n";
    cout << "\n" << "heuristic(1, 2, 3, 4) = " << heuristic(1, 2, 3, 4) << "\n";
    cout << "Correct result: 4" << "\n";
    cout << "\n";
  } else if (heuristic(2, -1, 4, -7) != 8) {
    cout << "testHeuristic Failed" << "\n";
    cout << "\n" << "heuristic(2, -1, 4, -7) = " << heuristic(2, -1, 4, -7) << "\n";
    cout << "Correct result: 8" << "\n";
    cout << "\n";
  } else {
    cout << "passed" << "\n";
  }
  return;
}

void testAddToOpen() {
  cout << "----------------------------------------------------------" << "\n";
  cout << "addToOpen Function test: ";
  int x = 3;
  int y = 0;
  int g = 5;
  int h = 7;
  vector<vector<int>> open{{0, 0, 2, 9}, {1, 0, 2, 2}, {2, 0, 2, 4}};
  vector<vector<int>> solution_open = open;
  solution_open.push_back(vector<int>{3, 0, 5, 7});
  vector<vector<State>> grid{{State::kClosed, State::kObstacle, State::kEmpty, State::kEmpty, State::kEmpty, State::kEmpty},
                            {State::kClosed, State::kObstacle, State::kEmpty, State::kEmpty, State::kEmpty, State::kEmpty},
                            {State::kClosed, State::kObstacle, State::kEmpty, State::kEmpty, State::kEmpty, State::kEmpty},
                            {State::kEmpty, State::kObstacle, State::kEmpty, State::kEmpty, State::kEmpty, State::kEmpty},
                            {State::kEmpty, State::kEmpty, State::kEmpty, State::kEmpty, State::kObstacle, State::kEmpty}};
  vector<vector<State>> solution_grid = grid;
  solution_grid[3][0] = State::kClosed;
  addToOpen(x, y, g, h, open, grid);
  if (open != solution_open) {
    cout << "failed" << "\n";
    cout << "\n";
    cout << "Your open list is: " << "\n";
    printVectorOfVectors(open);
    cout << "Solution open list is: " << "\n";
    printVectorOfVectors(solution_open);
    cout << "\n";
  } else if (grid != solution_grid) {
    cout << "failed" << "\n";
    cout << "\n";
    cout << "Your grid is: " << "\n";
    printVectorOfVectors(grid);
    cout << "\n";
    cout << "Solution grid is: " << "\n";
    printVectorOfVectors(solution_grid);
    cout << "\n";
  } else {
    cout << "passed" << "\n";
  }
  return;
}

void testCompare() {
  cout << "----------------------------------------------------------" << "\n";
  cout << "compare Function test: ";
  vector<int> test_1 {1, 2, 5, 6};
  vector<int> test_2 {1, 3, 5, 7};
  vector<int> test_3 {1, 2, 5, 8};
  vector<int> test_4 {1, 3, 5, 7};
  if (compare(test_1, test_2)) {
    cout << "failed" << "\n";
    cout << "\n" << "a = ";
    printVector(test_1);
    cout << "b = ";
    printVector(test_2);
    cout << "compare(a, b): " << compare(test_1, test_2) << "\n";
    cout << "Correct answer: 0" << "\n";
    cout << "\n";
  } else if (!compare(test_3, test_4)) {
    cout << "failed" << "\n";
    cout << "\n" << "a = ";
    printVector(test_3);
    cout << "b = ";
    printVector(test_4);
    cout << "compare(a, b): " << compare(test_3, test_4) << "\n";
    cout << "Correct answer: 1" << "\n";
    cout << "\n";
  } else {
    cout << "passed" << "\n";
  }
  return;
}

void testSearch() {
  cout << "----------------------------------------------------------" << "\n";
  cout << "search Function test: ";
  int init[2]{0, 0};
  int goal[2]{4, 5};
  auto board = readBoardFile("1.board");

  std::cout.setstate(std::ios_base::failbit); // Disable cout
  auto output = search(board, init, goal);
  std::cout.clear(); // Enable cout

  vector<vector<State>> solution{{State::kStart, State::kObstacle, State::kEmpty, State::kEmpty, State::kEmpty, State::kEmpty},
                            {State::kPath, State::kObstacle, State::kEmpty, State::kEmpty, State::kEmpty, State::kEmpty},
                            {State::kPath, State::kObstacle, State::kEmpty, State::kClosed, State::kClosed, State::kClosed},
                            {State::kPath, State::kObstacle, State::kClosed, State::kPath, State::kPath, State::kPath},
                            {State::kPath, State::kPath, State::kPath, State::kPath, State::kObstacle, State::kFinish}};

  if (output != solution) {
    cout << "failed" << "\n";
    cout << "search(board, {0,0}, {4,5})" << "\n";
    cout << "Solution board: " << "\n";
    printVectorOfVectors(solution);
    cout << "Your board: " << "\n";
    printVectorOfVectors(output);
    cout << "\n";
  } else {
    cout << "passed" << "\n";
  }
  return;
}

void testCheckValidCell() {
  cout << "----------------------------------------------------------" << "\n";
  cout << "checkValidCell Function test: ";
  vector<vector<State>> grid{{State::kClosed, State::kObstacle, State::kEmpty, State::kEmpty, State::kEmpty, State::kEmpty},
                            {State::kClosed, State::kObstacle, State::kEmpty, State::kEmpty, State::kEmpty, State::kEmpty},
                            {State::kClosed, State::kObstacle, State::kEmpty, State::kEmpty, State::kEmpty, State::kEmpty},
                            {State::kClosed, State::kObstacle, State::kEmpty, State::kEmpty, State::kEmpty, State::kEmpty},
                            {State::kClosed, State::kClosed, State::kEmpty, State::kEmpty, State::kObstacle, State::kEmpty}};

  if (checkValidCell(0, 0, grid)) {
    cout << "failed" << "\n";
    cout << "\n" << "test grid is: " << "\n";
    printVectorOfVectors(grid);
    cout << "Cell checked: (0, 0)" << "\n";
    cout << "\n";
  } else if (!checkValidCell(4, 2, grid)) {
    cout << "failed" << "\n";
    cout << "\n" << "test grid is: " << "\n";
    printVectorOfVectors(grid);
    cout << "Cell checked: (4, 2)" << "\n";
    cout << "\n";
  } else {
    cout << "passed" << "\n";
  }
}

void testExpandNeighbors() {
  cout << "----------------------------------------------------------" << "\n";
  cout << "expandNeighbosr Function test: ";
  vector<int> current{4, 2, 7, 3};
  int goal[2] {4, 5};
  vector<vector<int>> open{{4, 2, 7, 3}};
  vector<vector<int>> solution_open = open;
  solution_open.push_back(vector<int>{3, 2, 8, 4});
  solution_open.push_back(vector<int>{4, 3, 8, 2});
  vector<vector<State>> grid{{State::kClosed, State::kObstacle, State::kEmpty, State::kEmpty, State::kEmpty, State::kEmpty},
                            {State::kClosed, State::kObstacle, State::kEmpty, State::kEmpty, State::kEmpty, State::kEmpty},
                            {State::kClosed, State::kObstacle, State::kEmpty, State::kEmpty, State::kEmpty, State::kEmpty},
                            {State::kClosed, State::kObstacle, State::kEmpty, State::kEmpty, State::kEmpty, State::kEmpty},
                            {State::kClosed, State::kClosed, State::kEmpty, State::kEmpty, State::kObstacle, State::kEmpty}};
  vector<vector<State>> solution_grid = grid;
  solution_grid[3][2] = State::kClosed;
  solution_grid[4][3] = State::kClosed;
  expandNeighbors(current, open, grid, goal);
  cellSort(&open);
  cellSort(&solution_open);
  if (open != solution_open) {
    cout << "failed" << "\n";
    cout << "\n";
    cout << "Your open list is: " << "\n";
    printVectorOfVectors(open);
    cout << "Solution open list is: " << "\n";
    printVectorOfVectors(solution_open);
    cout << "\n";
  } else if (grid != solution_grid) {
    cout << "failed" << "\n";
    cout << "\n";
    cout << "Your grid is: " << "\n";
    printVectorOfVectors(grid);
    cout << "\n";
    cout << "Solution grid is: " << "\n";
    printVectorOfVectors(solution_grid);
    cout << "\n";
  } else {
  	cout << "passed" << "\n";
  }
  cout << "----------------------------------------------------------" << "\n";
  return;
}

int main()
{
    testHeuristic();
    testAddToOpen();
    testCompare();
    testSearch();
    testCheckValidCell();
    testExpandNeighbors();
 
}
