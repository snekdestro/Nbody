#ifndef particle_h
#define particle_h


namespace Body{
 
__global__ void compute_gravity(float* x, float* y, float* m, float* ax, float* ay, int n);
__global__ void move(float* d_ptr,float* x, float* y, float* ax, float* ay, float* vx, float* vy, int n, float dt); 
__global__ void compute_electric(float* x, float* y, float* q, float* ax, float* ay, int n);
__global__ void compute_gfield(float* x, float* y, float* m, float p_x, float p_y, int n, float* output);
__global__ void compute_efield(float* x, float* y, float* q, float p_x, float p_y, int n, float* output);
}

#endif