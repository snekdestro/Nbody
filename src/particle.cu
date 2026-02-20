#include "particle.cuh"
#include <cmath>
using namespace Body;


__global__ void Body::compute_gravity(particle2D* parts, int n,float soft){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float my_x = (i < n) ? parts[i].x : 0.0f;
    float my_y = (i < n) ? parts[i].y : 0.0f;
    float fx = 0.0f;
    float fy = 0.0f;
    __shared__ float sh_x[MAX_THREADS_PER_BLOCK];
    __shared__ float sh_y[MAX_THREADS_PER_BLOCK];
    __shared__ float sh_m[MAX_THREADS_PER_BLOCK];
    for(int tile = 0; tile < gridDim.x; tile++){
        int idx = tile * blockDim.x + threadIdx.x;
        if(idx < n){
            sh_x[threadIdx.x] = parts[idx].x;
            sh_y[threadIdx.x] = parts[idx].y;
            sh_m[threadIdx.x] = parts[idx].mass;
        }
        else{
            sh_x[threadIdx.x] = 0.0f;
            sh_y[threadIdx.x] = 0.0f;
            sh_m[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        #pragma unroll 
        for (int j = 0; j < MAX_THREADS_PER_BLOCK; j++) {

            float dx = sh_x[j] - my_x;
            float dy = sh_y[j] - my_y;
            float dist_sq = dx * dx + dy * dy + soft;
            float inv_dist = rsqrtf(dist_sq);
            float s = sh_m[j] * inv_dist * inv_dist * inv_dist ; 
            fx += dx * s;
            fy += dy * s;
        }
        __syncthreads();
    }
    if(i < n){
        parts[i].ax = fx;
        parts[i].ay = fy;
    }
}
__global__ void Body::compute_electric(particle2D* parts, int n,float soft){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float my_x = (i < n) ? parts[i].x : 0.0f;
    float my_y = (i < n) ? parts[i].y : 0.0f;
    float q_cur = (i < n) ? parts[i].charge : 0.0f;
    float fx = 0.0f;
    float fy = 0.0f;
    __shared__ float sh_x[MAX_THREADS_PER_BLOCK];
    __shared__ float sh_y[MAX_THREADS_PER_BLOCK];
    __shared__ float sh_q[MAX_THREADS_PER_BLOCK];
    for(int tile = 0; tile < gridDim.x; tile++){
        int idx = tile * blockDim.x + threadIdx.x;
        if(idx < n){
            sh_x[threadIdx.x] = parts[idx].x;
            sh_y[threadIdx.x] = parts[idx].y;
            sh_q[threadIdx.x] = parts[idx].charge;
        }
        else{
            sh_x[threadIdx.x] = 0.0f;
            sh_y[threadIdx.x] = 0.0f;
            sh_q[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        #pragma unroll 
        for (int j = 0; j < MAX_THREADS_PER_BLOCK; j++) {
 
            float dx = sh_x[j] - my_x;
            float dy = sh_y[j] - my_y;
            float dist_sq = dx * dx + dy * dy + soft;
            float inv_dist = rsqrtf(dist_sq);
            float s = inv_dist * inv_dist * inv_dist * sh_q[j] ; 
            fx += dx * s;
            fy += dy * s;
        }
        __syncthreads();
    }
    if(i < n){
        parts[i].ax = fx * -q_cur;
        parts[i].ay = fy * -q_cur;
    }
}

__global__ void Body::move(float* d_ptr, particle2D* parts, int n, float dt){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        parts[i].x = parts[i].x + parts[i].vx * dt + 0.5f * parts[i].ax * dt * dt; 
        parts[i].y = parts[i].y + parts[i].vy * dt + 0.5f * parts[i].ay * dt * dt; 
        d_ptr[i * 6] = parts[i].x;
        d_ptr[i * 6 + 1] = parts[i].y;
        parts[i].vx += parts[i].ax * dt;
        parts[i].vy += parts[i].ay * dt;
        float deg = atan2f(parts[i].vx,parts[i].vy);
        //d_ptr[i * 5 + 2] = sin(deg);
        //d_ptr[i * 5 + 3] = sin(deg + 2.0f * 3.141592653589f  / 3.0f);
        //d_ptr[i * 5 + 4] = sin(deg + 4.0f * 3.141592653589f  / 3.0f);
        //needs smoothing
        d_ptr[i * 6  +2] = 0.5 - 0.5 *powf(1.00001f, -(parts[i].vx * parts[i].vx + parts[i].vy * parts[i].vy));
        d_ptr[i * 6 + 3] = 0.0f;
        d_ptr[i * 6 + 4] = powf(1.00001f, -(parts[i].vx * parts[i].vx + parts[i].vy * parts[i].vy));
        d_ptr[i * 6 + 5] = parts[i].attrib;
    }
}

__global__ void Body::compute_gfield(particle2D* parts, float p_x, float p_y, int n,float* output,float soft){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        float lx =0.0f;
        float ly =0.0f;
        if(i < n){
            float dx = parts[i].x - p_x;
            float dy = parts[i].y - p_y;
            float dist_sq = dx * dx + dy * dy + soft;
            float invdist = rsqrt(dist_sq);
            float s = invdist * invdist * invdist * 1e-6f * parts[i].mass;
            lx += dx * s;
            ly += dy * s;

        }
        __shared__ float fx[MAX_THREADS_PER_BLOCK];
        __shared__ float fy[MAX_THREADS_PER_BLOCK];
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

__global__ void Body::compute_efield(particle2D* parts, float p_x, float p_y, int n,float* output,float soft){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
        float lx =0.0f;
        float ly =0.0f;
        if(i < n){
            float dx = parts[i].x - p_x;
            float dy = parts[i].y - p_y;
            float dist_sq = dx * dx + dy * dy + soft;
            float invdist = rsqrt(dist_sq);
            float s = invdist * invdist * invdist * 1e-6f * -parts[i].charge;
            lx += dx * s;
            ly += dy * s;

        }
        __shared__ float fx[MAX_THREADS_PER_BLOCK];
        __shared__ float fy[MAX_THREADS_PER_BLOCK];
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

__global__ void Body::compute_gravity3D(particle3D* parts, int n,float soft){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double my_x = (i < n) ? parts[i].x : 0.0;
    double my_y = (i < n) ? parts[i].y : 0.0;
    double my_z = (i < n) ? parts[i].z : 0.0;
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    __shared__ double sh_x[MAX_THREADS_PER_BLOCK];
    __shared__ double sh_y[MAX_THREADS_PER_BLOCK];
    __shared__ double sh_z[MAX_THREADS_PER_BLOCK];
    __shared__ double sh_m[MAX_THREADS_PER_BLOCK];
    for(int tile = 0; tile < gridDim.x; tile++){
        int idx = tile * blockDim.x + threadIdx.x;
        if(idx < n){
            sh_x[threadIdx.x] = parts[idx].x;
            sh_y[threadIdx.x] = parts[idx].y;
            sh_z[threadIdx.x] = parts[idx].z;
            sh_m[threadIdx.x] = parts[idx].mass;
        }
        else{
            sh_x[threadIdx.x] = 0.0;
            sh_y[threadIdx.x] = 0.0;
            sh_z[threadIdx.x] = 0.0;
            sh_m[threadIdx.x] = 0.0;
        }
        __syncthreads();
        #pragma unroll 
        for (int j = 0; j < MAX_THREADS_PER_BLOCK; j++) {

            double dx = sh_x[j] - my_x;
            double dy = sh_y[j] - my_y;
            double dz = sh_z[j] - my_z;
            double dist_sq = dx * dx + dy * dy + dz * dz + soft;
            double inv_dist = rsqrtf(dist_sq);
            double s = sh_m[j] * inv_dist * inv_dist * inv_dist ; 
            fx += dx * s;
            fy += dy * s;
            fz += dz * s;
        }
        __syncthreads();
    }
    if(i < n){
        parts[i].ax = fx;
        parts[i].ay = fy;
        parts[i].az = fz;
    }
}
__global__ void Body::move(float* d_ptr,particle3D* parts,int n, float dt,float){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        parts[i].x = parts[i].x + parts[i].vx * dt + 0.5f * parts[i].ax * dt * dt; 
        parts[i].y = parts[i].y + parts[i].vy * dt + 0.5f * parts[i].ay * dt * dt; 
        parts[i].z = parts[i].z + parts[i].vz * dt + 0.5f * parts[i].az * dt * dt; 
        parts[i].vx += parts[i].ax * dt;
        parts[i].vy += parts[i].ay * dt;
        parts[i].vz += parts[i].az * dt;
        d_ptr[i * 7] = parts[i].x;
        d_ptr[i * 7 + 1] = parts[i].y;
        d_ptr[i * 7 + 2] = parts[i].z;
        d_ptr[i * 7 + 3] = 1.0;
        d_ptr[i * 7 + 4] = 0.0;
        d_ptr[i * 7 + 5] = 0.0;
        d_ptr[i * 7 + 6] = parts[i].mass;
    }
}
