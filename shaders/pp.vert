#version 330 core
layout (location = 0) in vec2 aPos;   // Matches Attribute 0 (stride offset 0)
layout (location = 1) in vec3 aColor; // Matches Attribute 1 (stride offset 2 floats)

out vec3 vColor; // This "out" variable sends the color to the fragment shader

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0); 
    vColor = aColor; // Pass the color through
}