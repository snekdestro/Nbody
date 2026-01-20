#version 330 core
in vec3 vColor; // Received from the vertex shader
out vec4 FragColor;

void main() {
    // Output the color with full opacity (1.0)
    FragColor = vec4(vColor, 1.0);
}