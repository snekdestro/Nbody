#ifndef particle_cuh
#define particle_cuh
#define MAX_THREADS_PER_BLOCK 512


typedef struct{
    float x;
    float y;
    float z;
    float ax;
    float ay;
    float az;
    float vx;
    float vy;
    float vz;
    float mass;
    float charge;

} particle3D;

typedef struct{
    float x;
    float y;
    float ax;
    float ay;
    float vx;
    float vy;
    float mass;
    float charge;
    float attrib;
}particle2D;


namespace Body{
 
__global__ void compute_gravity(particle2D* parts, int n,float soft);
__global__ void move(float* d_ptr,particle2D* parts, int n, float dt); 
__global__ void compute_electric(particle2D* parts, int n,float soft);
__global__ void compute_gfield(particle2D* parts, float p_x, float p_y, int n, float* output,float soft);
__global__ void compute_efield(particle2D* parts, float p_x, float p_y, int n, float* output,float soft);
__global__ void compute_gravity3D(particle3D* parts, int n,float soft);
__global__ void move(float* d_ptr,particle3D* parts,int n, float dt,float); 

}

#endif