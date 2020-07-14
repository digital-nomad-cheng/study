//
//  shader.hpp
//  LearnOpenGL
//
//  Created by yuhua.cheng on 2020/7/14.
//  Copyright © 2020 idealabs. All rights reserved.
//

#ifndef shader_hpp
#define shader_hpp

#include <glad/glad.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class Shader
{
public:
    unsigned int ID;
    Shader(const GLchar *vertex_path, const GLchar *fragment_path);
    void use();
    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;
};

#endif /* shader_hpp */
