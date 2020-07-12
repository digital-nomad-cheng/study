## Config Project

1. Download glfw [here](https://www.glfw.org/download.html)
2. Compile glfw
3. Create XCode C++ project, link glfw library.	
	```C++
	#include <GLFW/glfw3.h>
	```
4. Configure [GLAD](https://glad.dav1d.de/) according to [tutorial](https://learnopengl-cn.github.io/01%20Getting%20started/02%20Creating%20a%20window/). Then link glad:
	```C++
	// include glad before glfw otherwise there will be compile error.
	#include <glad/glad.h> 
	#include <GLFW/glfw3.h>
	```
5. Add `IOKit.framework` and `Cocoa.framework` to project.


## Reference

1. Learn OpenGL: https://learnopengl-cn.github.io/