#include "particle.h"
#include <cmath>
using namespace Body;

void G_particle::calc(G_particle other){
    float r = std::sqrt((other.x - x) * (other.x - x) + (other.y - y) * (other.y - y));
    a_x += (other.x - x) / (r * r * r); 
    a_y += (other.y - y) / (r * r * r); 

}

void G_particle::move(float dt){
    x += v_x * dt + 0.5f * a_x * dt * dt;
    y += v_y * dt + 0.5f * a_y * dt * dt;
    a_x = 0.0f;
    a_y = 0.0f;
}
__global__ void Body::compute_gravity(float* x, float* y, float* m, float* ax, float* ay, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float my_x = (i < n) ? x[i] : 0.0f;
    float my_y = (i < n) ? y[i] : 0.0f;
    float fx = 0.0f;
    float fy = 0.0f;
    __shared__ float sh_x[256];
    __shared__ float sh_y[256];
    for(int tile = 0; tile < gridDim.x; tile++){
        int idx = tile * blockDim.x + threadIdx.x;
        if(idx < n){
            sh_x[threadIdx.x] = x[idx];
            sh_y[threadIdx.x] = y[idx];
        }
        else{
            sh_x[threadIdx.x] = 0.0f;
            sh_y[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        #pragma unroll 
        for (int j = 0; j < 256; j++) {
            float dx = sh_x[j] - my_x;
            float dy = sh_y[j] - my_y;
            float dist_sq = dx * dx + dy * dy + 1e-9f;
            float inv_dist = rsqrtf(dist_sq);
            float s = inv_dist * inv_dist * inv_dist * 1e-7f; 
            fx += dx * s;
            fy += dy * s;
        }
        __syncthreads();
    }
    if(i < n){
        ax[i] = fx;
        ay[i] = fy;
    }







}

__global__ void Body::move(float* d_ptr, float* x, float* y, float* ax, float* ay, float* vx, float* vy, int n, float dt){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        d_ptr[i * 5] = d_ptr[i * 5] + vx[i] * dt + 0.5f * ax[i] * dt * dt; 
        d_ptr[i * 5 + 1] = d_ptr[i * 5 + 1] + vy[i] * dt + 0.5f * ay[i] * dt * dt; 
        x[i * 5] = d_ptr[i * 5];
        y[i * 5 + 1] = d_ptr[i * 5 + 1];
        d_ptr[i * 5 + 2] = sqrtf(vx[i] * vx[i] + vy[i] * vy[i]); // Red
        d_ptr[i * 5 + 3] = 0.5f;         // Green
        d_ptr[i * 5 + 4] = 1.0f;         // Blue
        vx[i] += ax[i];
        vy[i] += ay[i];
    }
}


