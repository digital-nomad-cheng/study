/**
 * File: life.cpp
 * --------------
 * Implements the Game of Life.
 */

#include <iostream>  // for cout
#include <string>
#include <fstream>
#include <chrono>
#include <thread>

using namespace std;

#include "console.h" // required of all files that contain the main function
#include "simpio.h"  // for getLine
#include "gevents.h" // for mouse event detection
#include "strlib.h"

#include "life-constants.h"  // for kMaxAge
#include "life-graphics.h"   // for class LifeDisplay

/**
 * Function: welcome
 * -----------------
 * Introduces the user to the Game of Life and its rules.
 */
static void welcome() {
    cout << "Welcome to the game of Life, a simulation of the lifecycle of a bacteria colony." << endl;
    cout << "Cells live and die by the following rules:" << endl << endl;
    cout << "\tA cell with 1 or fewer neighbors dies of loneliness" << endl;
    cout << "\tLocations with 2 neighbors remain stable" << endl;
    cout << "\tLocations with 3 neighbors will spontaneously create life" << endl;
    cout << "\tLocations with 4 or more neighbors die of overcrowding" << endl << endl;
    cout << "In the animation, new cells are dark and fade to gray as they age." << endl << endl;
    getLine("Hit [enter] to continue....   ");
}

static void parseFileIntoGrid(std::string& file_name, Grid<int>& grid)
{
    if (file_name.empty()) {
        std::cout << "Random initialize grid." << std::endl;
        int width = std::rand() % 21 + 40;
        int height = std::rand() % 21 + 40;
        grid.resize(height, width);
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                if (std::rand() % 2) {
                    int age = std::rand() % (kMaxAge) + 1;
                    grid[h][w] = age;
                }
            }
        }
    } else {
        std::cout << "Initialize from file:" << file_name << std::endl;
        std::ifstream file(file_name);
        if (!file.is_open()) {
            std::cout << "Failed to open file." << std::endl;
            return;
        }
        std::string line;
        int line_count = 0;
        int width = 0;
        int height = 0;
        int h = 0;
        int w = 0;
        while(std::getline(file, line)) {
            // std::cout << line << std::endl;
            if (line[0] == '#') {
                continue;
            } else if(line_count == 0) {
                height = stringToInteger(line);
                line_count += 1;
            } else if(line_count == 1) {
                width = stringToInteger(line);
                line_count += 1;
                grid.resize(height, width);
            } else {
                w = 0;
                for (char& c : line) {
                    if (c == 'X') {
                        grid[h][w] = 1;
                    }
                    w += 1;
                }
                h += 1;
            }
        }
    }
}

enum SimulationMode {
    Fast = 1,
    Medium,
    Slow,
    Manual,
};

SimulationMode chooseSimulationMode()
{
   std::string mode = getLine("You choose how fast to run the simulation.\n"
           "1 = As fast as this chip can go!\n"
           "2 = Not too fast, this is a school zone.\n"
           "3 = Nice and slow so I can watch everything that happens.\n"
           "4 = Require enter key be pressed before advancing to next generation.\n");
    if (stringToInteger(mode) == 1) {
        return SimulationMode::Fast;
    } else if(stringToInteger(mode) == 2) {
        return SimulationMode::Medium;
    } else if(stringToInteger(mode) == 3) {
        return SimulationMode::Slow;
    } else if(stringToInteger(mode) == 4) {
        return SimulationMode::Manual;
    } else {
        std::cout << "Invalide input, please choose agian!" << std::endl;
        return chooseSimulationMode();
    }
}

void displayCells(LifeDisplay& display, Grid<int>& cells)
{
    for (int h = 0; h < cells.numRows(); h++) {
        for (int w = 0; w < cells.numCols(); w++) {
            display.drawCellAt(h, w, cells[h][w]);
        }
    }
}

int numOfNeighbors(Grid<int>& cells, int h, int w)
{



    return (cells[h-1][w-1] > 0) +
           (cells[h-1][w] > 0) +
           (cells[h-1][w+1] >0) +
           (cells[h][w-1] > 0) +
           (cells[h][w+1] > 0) +
           (cells[h+1][w-1] > 0) +
           (cells[h+1][w] > 0) +
           (cells[h+1][w+1] > 0);

}

void simulate(Grid<int>& current_cells, Grid<int>& next_cells)
{
    next_cells.resize(current_cells.numRows(), current_cells.numCols());
    Grid<int> padded_cells = Grid<int>(current_cells.numRows()+2, current_cells.numCols()+2);
    for (int h = 0; h < current_cells.numRows(); h++) {
        for (int w = 0; w < current_cells.numCols(); w++) {
            padded_cells[h+1][w+1] = current_cells[h][w];
        }
    }

    for (int h = 0; h < current_cells.numRows(); h++)
        for(int w = 0; w < current_cells.numCols(); w++)
            next_cells[h][w] = current_cells[h][w];

    for (int h = 0; h < current_cells.numRows(); h++) {
        for(int w = 0; w < current_cells.numCols(); w++) {
            int num_neighbors = numOfNeighbors(padded_cells, h+1, w+1);
            if (num_neighbors <= 1) {
                next_cells[h][w] = 0;
            } else if(num_neighbors == 2) {
                if (current_cells[h][w] == 0) {
                    next_cells[h][w] = 0;
                } else {
                    next_cells[h][w] += 1;
                }
            } else if (num_neighbors == 3) {
                next_cells[h][w] += 1;
            } else if (num_neighbors >= 4) {
                next_cells[h][w] = 0;
            }
        }
    }

    for (int h = 0; h < current_cells.numRows(); h++)
        for(int w = 0; w < current_cells.numCols(); w++)
            current_cells[h][w] = next_cells[h][w];
}


/**
 * Function: main
 * --------------
 * Provides the entry point of the entire program.
 */
int main() {
    LifeDisplay display;
    display.setTitle("Game of Life");
    welcome();
    Grid<int> current_cells;
    Grid<int> next_cells;

    std::string file_name = getLine("You can start your colony with random cells or read from a prepared file.\n"
            "Enter name of colony file (or RETURN to seed randomly):");
    parseFileIntoGrid(file_name, current_cells);
    int height = current_cells.numRows();
    int width = current_cells.numCols();
    display.setDimensions(height, width);
    displayCells(display, current_cells);
    display.repaint();
    SimulationMode mode = chooseSimulationMode();


    while (1) {
        switch(mode) {
            case SimulationMode::Fast:
                simulate(current_cells, next_cells);
                displayCells(display, current_cells);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                display.repaint();
                break;
            case SimulationMode::Medium:
                simulate(current_cells, next_cells);
                displayCells(display, current_cells);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                display.repaint();
                break;
            case SimulationMode::Slow:
                simulate(current_cells, next_cells);
                displayCells(display, current_cells);
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                display.repaint();
                break;
            default:
                break;
        }
    }
    return 0;
}
