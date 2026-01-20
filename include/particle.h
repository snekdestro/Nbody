#ifndef particle_h
#define particle_h


namespace Body{
    class G_particle{
        public:
            float x;
            float y;
            float v_x;
            float v_y; 
            float a_x;
            float a_y;
            int m;
            void calc(G_particle other);
            void move(float dt);
    };
__global__ void compute_gravity(float* x, float* y, float* m, float* ax, float* ay, int n);
__global__ void move(float* d_ptr,float* x, float* y, float* ax, float* ay, float* vx, float* vy, int n, float dt); 
}

#endif