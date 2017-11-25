
#include "common.hpp"
#include "util.hpp"

using namespace std;
using namespace glm;
using namespace agp;

GLuint VAO = 0;
GLuint shaderProgram = 0;


void init()
{
    // set up vertex data (and buffer(s)) and configure vertex attributes
  	// ------------------------------------------------------------------
	glGenVertexArrays(1, &VAO);

	shaderProgram = util::loadShaders("./vertexShader.glsl", "./fragmentShader.glsl");
	
  	glUseProgram(shaderProgram);

    // Set the background color (RGBA), change to color black
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Your OpenGL settings, such as alpha, depth and others, should be
    // defined here! For the assignment, we only ask you to enable the
    // alpha channel.
}

void release()
{

    // de-allocate all resources once they've outlived their purpose:
  	// ------------------------------------------------------------------------
  	glDeleteVertexArrays(1, &VAO);
    
    // Do not forget to release any memory allocation here!
}

void display()
{
    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  	glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
    
    glutSolidSphere(1, 20, 20);

 	//   printf("FreeGLUT triggered the display() callback!\n");
    
    // Your rendering code must be here! Do not forget to swap the
    // front and back buffers, and to force a redisplay to keep the
    // render loop running. This functionality is available within
    // FreeGLUT as well, check the assignment for more information.
    
    // Important note: The following function flushes the rendering
    // queue, but this is only for single-buffered rendering. You
    // must replace this function following the previous indications.
    
    // glFlush(); //used for single-buffered rendering
    // Replacing single-buffered rendering with double-buffered rendering
    glutSwapBuffers();
    // Force the rendering loop to remain active
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
    // Capture the keystrokes of your keyboard and print
    printf("Keystroke captured is: %c \n", key);
    if (key == 0x1B)
    {
    	printf("ESC captured, the application is going to be closed! \n");
    	// Release all the allocated memory
    	release();
    	// Exit the program
    	exit(0);
    }
    
}


int main(int argc, char **argv)
{
    // Initialize FreeGLUT and create the window
    glutInit(&argc, argv);
    
    // Setup the window (e.g., size, display mode and so on)
    glutInitWindowSize(1280, 720);
    // glutInitWindowPosition( ... );
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA); //Initialized display mode to enable double buffering and RGBA
    
    // Make FreeGLUT return from the main rendering loop
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                  GLUT_ACTION_GLUTMAINLOOP_RETURNS);

    // Create the window and associate the callbacks
    glutCreateWindow("Applied GPU Programming");
    glutDisplayFunc(display);
    // glutIdleFunc( ... );
    // glutReshapeFunc( ... );
    glutKeyboardFunc(keyboard); 	//Register keyboard function
    // glutSpecialFunc( ... );
    // glutMouseFunc( ... );
    // glutMotionFunc( ... );
    
    // Init GLAD to be able to access the OpenGL API
    if (!gladLoadGL())
    {
        return GL_INVALID_OPERATION;
    }
    
    // Display OpenGL information
    util::displayOpenGLInfo();

    
    // Initialize the 3D view
    init();
    
    // Launch the main loop for rendering
    glutMainLoop();

    
     
	return 0;
}

