/**
 * File: life.cpp
 * --------------
 * Implements the Game of Life.
 */

#include <iostream>  // for cout
#include <string>
#include <fstream>
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

static void parseFileIntoGrid(std::string& file_name, LifeDisplay& display)
{
    if (file_name.empty()) {
        std::cout << "Random initialize grid." << std::endl;
        int width = std::rand() % 21 + 40;
        int height = std::rand() % 21 + 40;
        display.setDimensions(height, width);
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                if (std::rand() % 2) {
                    int age = std::rand() % (kMaxAge) + 1;
                    display.drawCellAt(h, w, age);
                }
            }
        }
    } else {
        std::cout << "Initialize from file:" << file_name << std::endl;
        std::ifstream file(file_name);
        if (!file.is_open()) {
            std::cout << "Failed to open file" << std::endl;
            return;
        }
        std::string line;
        int width = 0;
        int height = 0;
        while(std::getline(file, line)) {
            // std::cout << line << std::endl;
            if (line[0] == '#') {
                continue;
            } else {
                width = line.length();
                height += 1;
            }
        }
        std::cout << "width: " << width << "height: " << height << std::endl;
        display.setDimensions(height, width);
        file = std::ifstream(file_name);
        int h = 0;
        int w = 0;
        while(std::getline(file, line)) {
            // std::cout << line << std::endl;
            w = 0;
            if (line[0] == '#') {
                continue;
            } else {
                for (char& c : line) {
                    if (c == 'X') {
                        std::cout << c << std::endl;
                        display.drawCellAt(h, w, 1);
                    }
                    w += 1;
                }
                h += 1;
            }
        }
    }
//    while (1)
//        display.repaint();
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
    std::string file_name = getLine("You can start your colony with random cells or read from a prepared file.\n"
            "Enter name of colony file (or RETURN to seed randomly):");
    parseFileIntoGrid(file_name, display);


    return 0;
}
