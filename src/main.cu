#include <particle.h>
#include <glad/glad.h>
#include <gl/GL.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>

unsigned int WINDOW_WIDTH = 1080;
unsigned int WINDOW_HEIGHT = 1080;
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
int main(int argc, char const *argv[])
{   
    
    
    
    

    glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "OpenGL 4.6 Template", NULL, NULL);
	glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    const int N = 256 * 256 * 2;
    const int threads = 256;
    int blocks = (N + threads -1 )/ threads; 
    std::vector<float> x(N);
    std::vector<float> y(N);
    std::vector<float> m(N);
    std::vector<float> vx(N);
    std::vector<float> vy(N);
    srand(static_cast<unsigned int>(time(nullptr)));
    for(int i =0; i < N; i++){
        x[i] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX /2.0f) - 1.0f );
        y[i] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX /2.0f) - 1.0f  );
        vx[i] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX /2.0f) - 1.0f) * 10.0f;
        vy[i] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX /2.0f) - 1.0f) * 10.0f;
        //std::printf("%f, %f\n", x[i],y[i]); //debug print

        m[i] = 1.0;
    }
    float *d_x, *d_y, *d_vx, *d_vy, *d_ax, *d_ay, *d_m;
    cudaMalloc(&d_x,  N * sizeof(float));
    cudaMalloc(&d_y,  N * sizeof(float));
    cudaMalloc(&d_vx, N * sizeof(float));
    cudaMalloc(&d_vy, N * sizeof(float));
    cudaMalloc(&d_ax, N * sizeof(float));
    cudaMalloc(&d_ay, N * sizeof(float));
    cudaMalloc(&d_m,  N * sizeof(float));
    cudaMemcpy(d_x, &x[0], N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &y[0], N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, &m[0], N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, &vx[0], N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, &vy[0], N * sizeof(float), cudaMemcpyHostToDevice);
    
    //cudaMemset(d_vx, 0, N * sizeof(float));
    //cudaMemset(d_vy, 0, N * sizeof(float));
    cudaMemset(d_ax, 0, N * sizeof(float));
    cudaMemset(d_ay, 0, N * sizeof(float));
    unsigned int VAO, VBO, EBO;
    struct cudaGraphicsResource *cuda_vbo_resource;
    
    // Create OpenGL Buffer
    glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
    
    glGenBuffers(1, &VBO);
 
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, N * 5 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    


    
    // Attribute 0: Position (x, y)
    // Stride is 5 * sizeof(float) because each particle's data repeats every 5 floats
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Attribute 1: Color (r, g, b)
    // This starts at an offset of 2 floats (skipping x and y)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // Register Buffer with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsMapFlagsWriteDiscard);
    const char* vertexShaderSource = "#version 330 core\n"
        "layout (location = 0) in vec2 aPos;\n"
        "layout (location = 1) in vec3 aColor;\n"
        "out vec3 vColor;\n"
        "void main() {\n"
        "   gl_Position = vec4(aPos, 0.0, 1.0);\n"
        "   gl_PointSize = 10.0;\n"
        "   vColor = aColor;\n"
        "}\0";

    const char* fragmentShaderSource = "#version 330 core\n"
        "in vec3 vColor;\n"
        "out vec4 FragColor;\n"
        "void main() {\n"
        "   FragColor = vec4(vColor, 1.0);\n"
        "}\n\0";
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    // 3. Compile Fragment Shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // 4. Link Shaders into a Program
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // 5. Clean up individual shaders (they are linked now, no longer needed)
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    while (!glfwWindowShouldClose(window)) {
        float* d_ptr;
        size_t size;
		
        cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_ptr, &size, cuda_vbo_resource);


        Body::compute_gravity<<<blocks, threads>>>(d_x, d_y, d_m, d_ax, d_ay, N);
        Body::move<<<blocks, threads>>>(d_ptr ,d_x , d_y , d_ax, d_ay, d_vx, d_vy, N, 0.00001f); 
        cudaDeviceSynchronize();
        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

        glClear(GL_COLOR_BUFFER_BIT); // Clear the previous frame
    
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glUseProgram(shaderProgram); 
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, N);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

    return 0;
    
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
	WINDOW_WIDTH = width;
	WINDOW_HEIGHT = height;
}