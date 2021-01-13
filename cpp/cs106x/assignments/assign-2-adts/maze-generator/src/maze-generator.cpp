/**
 * File: maze-generator.cpp
 * ------------------------
 * Presents an adaptation of Kruskal's algorithm to generate mazes.
 */

#include <iostream>
#include <algorithm>
using namespace std;

#include "console.h"
#include "simpio.h"
#include "set.h"
#include "vector.h"
#include "maze-types.h"
#include "maze-graphics.h"

static int getMazeDimension(string prompt,
                            int minDimension = 7, int maxDimension = 50) {
    while (true) {
        int response = getInteger(prompt);
        if (response == 0) return response;
        if (response >= minDimension && response <= maxDimension) return response;
        cout << "Please enter a number between "
             << minDimension << " and "
             << maxDimension << ", inclusive." << endl;
    }
}

int main() {
    while (true) {
        int dimension = getMazeDimension("What should the dimension of your maze be [0 to exit]? ");
        if (dimension == 0) break;
        cout << "This is where I'd animate the construction of a maze of dimension " << dimension << "." << endl;
        MazeGeneratorView view;
        view.setDimension(dimension);
        view.drawBorder();
        view.repaint();

        // construct chambers
        Vector<cell> chambers;
        for (int h = 0; h < dimension; h++) {
            for (int w = 0; w < dimension; w++) {
                cell c{h, w};
                chambers.add(c);
            }
        }
        // construct walls
        Vector<wall> walls;
        for (int h = 0; h < dimension-1; h++) {
            for (int w = 0; w < dimension-1; w++) {
                wall w1{cell{h, w}, cell{h, w+1}};
                wall w2{cell{h, w}, cell{h+1, w}};
                walls.push_back(w1);
                walls.push_back(w2);
            }
        }
        for(int h = 0; h < dimension-1; h++) {
            wall w{cell{h, dimension-1}, cell{h+1, dimension-1}};
            walls.push_back(w);
        }
        for(int w = 0; w < dimension-1; w++) {
            wall wa{cell{dimension-1, w}, cell{dimension-1, w+1}};
            walls.push_back(wa);
        }
        view.addAllWalls(walls);
        view.repaint();

        // shuffle vectors
        std::random_shuffle(walls.begin(), walls.end());

        Set<Set<cell>> set_chambers;
        Set<cell> set_chamber;
        for (cell& c : chambers) {
            set_chambers.add(Set<cell>{c});
        }

        cout << chambers.size() << endl;
        while (chambers.size() > 1) {
            for (wall& w: walls) {
                bool same_chamber = false;
                for (auto it = set_chambers.begin(); it != set_chambers.end(); it++) {
                    set_chamber = *it;
                    if (set_chamber.contains(w.one) && set_chamber.contains(w.two)) {
                        // these two are in the same chamber
                        same_chamber = true;
                    }
                }
                if (!same_chamber) {

                    view.removeWall(w);
                    view.repaint();

                    // when they are in two seperate sets
                    Set<cell> merged_chamber;
                    Set<cell> chamber1;
                    Set<cell> chamber2;
                    for (Set<cell> set_chamber: set_chambers) {
                        // set_chamber = *it;
                        if (set_chamber.contains(w.two)) {
                            chamber2 = set_chamber;
                        }
                        if (set_chamber.contains(w.one)) {
                            chamber1 = set_chamber;
                        }
                    }
                    merged_chamber = chamber1 + chamber2;
                    set_chambers.remove(chamber1);
                    set_chambers.remove(chamber2);
                    set_chambers.add(merged_chamber);

                }
            }
         }

    }

    return 0;
}
