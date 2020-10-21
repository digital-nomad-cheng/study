## Config Project

1. Download glfw [here](https://www.glfw.org/download.html)
2. Compile glfw
3. Create XCode C++ project, link glfw library.	
	```C++
	#include <GLFW/glfw3.h>
	```
4. Configure [GLAD](https://glad.dav1d.de/) according to [tutorial](https://learnopengl-cn.github.io/01%20Getting%20started/02%20Creating%20a%20window/). Then link glad:
	GLAD is used to manage pointers of OpenGL, we must initialize glad before we call OpenGL functions.
	```C++
	// include glad before glfw otherwise there will be compile error.
	#include <glad/glad.h> 
	#include <GLFW/glfw3.h>
	```
5. Add `IOKit.framework` and `Cocoa.framework` to project.
	
	https://stackoverflow.com/questions/18391487/compiling-with-glfw3-linker-errors-undefined-reference



## Shaders
1. Shaders are very isolated programs  in that they'are not allowed to communicate with each other; the only communication they have
is via their inputs and outputs.
2. There are a maximum number of vertex attributes we'are allowed to decalure limited by the hardware. OpenGL speicification guarantees there are always at least 16 4-component vertex attributes available.
3. Shaders use `in` and `out` keyword for communication between shaders throughout pipelines.
4. `Unifroms` are another way to pass data from our application on the CPU to the shaders on the GPU besides `vertex attributes`. `Uniforms` are `global`. `Gloal` means two things: first a uniform variable is unique per shader program object, and can be accessed from any shader at any stage in the shader program; second, whatever you set the uniform value to, uniforms will keep their values until they'are either reset or updated.
5. glVertexAttribPointer 
```C++
	
	// vertex data on CPU
	loat vertices[] = {
		// location          // color
		 0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f,   // 右下
		-0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,   // 左下
		 0.0f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f    // 顶部
    };
 	// set vertex location attribute pointer
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // set vertex color attribute pointer
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
```
Parameters:
 1. The first parameter specifies which vertex attribute we want to configure. It matches the `layout (location = 0)` in vertex shader.
 2. The next parameter specifies the size of the vertex attribute. 
 3. The third parameter specifies the type of the data.
 4. The forth parameter specifies if we want the data to be normalized. If we're inputting integer data types (int, byte) and we've set this to `GL_TRUE`, the integer data is normalized to 0 (or -1 for signed data) and 1 when converted to float. 
 5. The fifth parameter is known as the stride and tells us the space between consecutive vertex attributes. we could've also specified the stride as 0 to let OpenGL determine the stride (this only works when values are tightly packed).
 6. The last parameter is of type void* and is the offset of where the position data begins in the buffer. 



## Reference

1. Learn OpenGL: https://learnopengl-cn.github.io/