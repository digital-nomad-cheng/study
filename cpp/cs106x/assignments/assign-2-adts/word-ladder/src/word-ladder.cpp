/**
 * File: word-ladder.cpp
 * ---------------------
 * Implements a program to find word ladders connecting pairs of words.
 */

#include <iostream>
using namespace std;

#include "console.h"
#include "lexicon.h"
#include "strlib.h"
#include "simpio.h"
#include "queue.h"
#include "vector.h"
#include "stack.h"
#include "set.h"

static string getWord(const Lexicon& english, const string& prompt) {
    while (true) {
        string response = trim(toLowerCase(getLine(prompt)));
        if (response.empty() || english.contains(response)) return response;
        cout << "Your response needs to be an English word, so please try again." << endl;
    }
}

static void generateNeighborWords(const Lexicon& english, const string& top, Set<string> used_words, Vector<string>& neighbors) {
    for (int i = 0; i < 26; i++) {
        char c = i + 'a';
        // cout << c << endl;
        for (int j = 0; j < top.size(); j++) {

            string neighbor = top;
            neighbor[j] = c;
            if (english.contains(neighbor) && !used_words.contains(neighbor)) {
                // this neighbor is a valid word and it has not been used before
                neighbors.push_back(neighbor);
            }

        }
    }
}


//static void generateValideNeighbor
static void generateLadder(const Lexicon& english, const string& start, const string& end) {
    cout << "Here's where you'll search for a word ladder connecting \"" << start << "\" to \"" << end << "\"." << endl;
    if (start.length() != end.length()) {
        cout << "The two endpoints must contain the same number of characters, or else no word ladder can exist." << endl;
        return;
    }

    Queue<Vector<string>> queue;
    Stack<string> result;
    result.push(start);
    Vector<string> middle_result;
    middle_result.push_back(start);
    queue.enqueue((middle_result));

    Set<string> used_words;
    used_words.add(start);

    while (!queue.isEmpty()) {
        Vector<string> front = queue.dequeue();
        int size = front.size();
        string top = front[size-1];
        if (top == end) {
            cout << "We found a solution:" << endl;
            for (string& str : front) {
                cout << str << " ";
            }
            cout << endl;
            return;
        } else {
            // top word not equal to end word, we extend the ladder.
            Vector<string> neighbors;
            generateNeighborWords(english, top, used_words, neighbors);
            for (string & neighbor : neighbors) {
                Vector<string> partial_ladder = front;
                partial_ladder.push_back(neighbor);
                used_words.add(neighbor);
                queue.enqueue(partial_ladder);
            }
        }

    }
}




static const string kEnglishLanguageDatafile = "dictionary.txt";
static void playWordLadder() {
    Lexicon english(kEnglishLanguageDatafile);
    while (true) {
        string start = getWord(english, "Please enter the source word [return to quit]: ");
        if (start.empty()) break;
        string end = getWord(english, "Please enter the destination word [return to quit]: ");
        if (end.empty()) break;
        generateLadder(english, start, end);
    }
}

int main() {
    cout << "Welcome to the CS106 word ladder application!" << endl << endl;
    playWordLadder();
    cout << "Thanks for playing!" << endl;
    return 0;
}
