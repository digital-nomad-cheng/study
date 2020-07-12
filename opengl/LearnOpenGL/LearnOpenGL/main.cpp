//
//  main.cpp
//  LearnOpenGL
//
//  Created by yuhua.cheng on 2020/7/12.
//  Copyright © 2020 idealabs. All rights reserved.
//

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

int main(int argc, const char * argv[]) {
    // insert code here...
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    std::cout << "Hello, World!\n";
    return 0;
}
