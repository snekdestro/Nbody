#include "particle.cuh"
#include <cmath>
using namespace Body;


__global__ void Body::compute_gravity(float* x, float* y, float* m, float* ax, float* ay, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float my_x = (i < n) ? x[i] : 0.0f;
    float my_y = (i < n) ? y[i] : 0.0f;
    float fx = 0.0f;
    float fy = 0.0f;
    __shared__ float sh_x[256];
    __shared__ float sh_y[256];
    __shared__ float sh_m[256];
    for(int tile = 0; tile < gridDim.x; tile++){
        int idx = tile * blockDim.x + threadIdx.x;
        if(idx < n){
            sh_x[threadIdx.x] = x[idx];
            sh_y[threadIdx.x] = y[idx];
            sh_m[threadIdx.x] = m[idx];
        }
        else{
            sh_x[threadIdx.x] = 0.0f;
            sh_y[threadIdx.x] = 0.0f;
            sh_m[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        #pragma unroll 
        for (int j = 0; j < 256; j++) {

            float dx = sh_x[j] - my_x;
            float dy = sh_y[j] - my_y;
            float dist_sq = dx * dx + dy * dy + 1e-9f;
            float inv_dist = rsqrtf(dist_sq);
            float s = sh_m[j] * inv_dist * inv_dist * inv_dist ; 
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
__global__ void Body::compute_electric(float* x, float* y,float* q, float* ax, float* ay, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float my_x = (i < n) ? x[i] : 0.0f;
    float my_y = (i < n) ? y[i] : 0.0f;
    float q_cur = (i < n) ? q[i] : 0.0f;
    float fx = 0.0f;
    float fy = 0.0f;
    __shared__ float sh_x[256];
    __shared__ float sh_y[256];
    __shared__ float sh_q[256];
    for(int tile = 0; tile < gridDim.x; tile++){
        int idx = tile * blockDim.x + threadIdx.x;
        if(idx < n){
            sh_x[threadIdx.x] = x[idx];
            sh_y[threadIdx.x] = y[idx];
            sh_q[threadIdx.x] = q[idx];
        }
        else{
            sh_x[threadIdx.x] = 0.0f;
            sh_y[threadIdx.x] = 0.0f;
            sh_q[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        #pragma unroll 
        for (int j = 0; j < 256; j++) {
            float qq = sh_q[j] * q_cur;
            float dx = sh_x[j] - my_x;
            float dy = sh_y[j] - my_y;
            float dist_sq = dx * dx + dy * dy + 1e-9f;
            float inv_dist = rsqrtf(dist_sq);
            float s = inv_dist * inv_dist * inv_dist ; 
            fx += dx * s * qq;
            fy += dy * s * qq;
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
        d_ptr[i * 5] = x[i] + vx[i] * dt + 0.5f * ax[i] * dt * dt; 
        d_ptr[i * 5 + 1] = y[i] + vy[i] * dt + 0.5f * ay[i] * dt * dt; 
        x[i] = d_ptr[i * 5];
        y[i] = d_ptr[i * 5 + 1];
        vx[i] += ax[i] * dt;
        vy[i] += ay[i] * dt;
        float deg = atan2f(vx[i],vy[i]);
        //d_ptr[i * 5 + 2] = sin(deg);
        //d_ptr[i * 5 + 3] = sin(deg + 2.0f * 3.141592653589f  / 3.0f);
        //d_ptr[i * 5 + 4] = sin(deg + 4.0f * 3.141592653589f  / 3.0f);
        //needs smoothing
        d_ptr[i * 5  +2] = 1 - powf(1.01f, -(vx[i] * vx[i] + vy[i] * vy[i]));
        d_ptr[i * 5 + 3] = 0.0f;
        d_ptr[i * 5 + 4] =  powf(1.01f, -(vx[i] * vx[i] + vy[i] * vy[i]));
        
    }
}

__global__ void Body::compute_gfield(float* x, float* y, float* m, float p_x, float p_y, int n,float* output){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        float lx =0.0f;
        float ly =0.0f;
        if(i < n){
            float dx = x[i] - p_x;
            float dy = y[i] - p_y;
            float dist_sq = dx * dx + dy * dy + 1e-9f;
            float invdist = rsqrt(dist_sq);
            float s = invdist * invdist * invdist * 1e-6f * m[i];
            lx += dx * s;
            ly += dy * s;

        }
        __shared__ float fx[256];
        __shared__ float fy[256];
        fx[threadIdx.x] = lx;
        fy[threadIdx.x] = ly;
        __syncthreads();
        for(unsigned int it = blockDim.x / 2; it > 0; it >>= 1){
            if(threadIdx.x < it){
                fx[threadIdx.x] += fx[threadIdx.x + it];
                fy[threadIdx.x] += fy[threadIdx.x + it];
            }
            __syncthreads();
        }
        if(threadIdx.x == 0){
            atomicAdd(&output[0], fx[0]);
            atomicAdd(&output[1], fy[0]);
        }
}

__global__ void Body::compute_efield(float* x, float* y, float* q, float p_x, float p_y, int n,float* output){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
        float lx =0.0f;
        float ly =0.0f;
        if(i < n){
            float dx = x[i] - p_x;
            float dy = y[i] - p_y;
            float dist_sq = dx * dx + dy * dy + 1e-9f;
            float invdist = rsqrt(dist_sq);
            float s = invdist * invdist * invdist * 1e-6f * q[i];
            lx += dx * s;
            ly += dy * s;

        }
        __shared__ float fx[256];
        __shared__ float fy[256];
        fx[threadIdx.x] = lx;
        fy[threadIdx.x] = ly;
        __syncthreads();
        for(unsigned int it = blockDim.x / 2; it > 0; it >>= 1){
            if(threadIdx.x < it){
                fx[threadIdx.x] += fx[threadIdx.x + it];
                fy[threadIdx.x] += fy[threadIdx.x + it];
            }
            __syncthreads();
        }
        if(threadIdx.x == 0){
            atomicAdd(&output[0], fx[0]);
            atomicAdd(&output[1], fy[0]);
        }
}

