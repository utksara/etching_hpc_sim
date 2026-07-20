#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include <math.h>

linear etch rate


#define NUM_THREADS 12
#define MAX_POINTS 1000
#define MIN_POINTS 10

float ** get_new_points(int N){
    float (*arr)[2] = malloc(N * sizeof(*arr));
    return arr;
}


float * get_new_array(int N){
    float *arr = malloc(N * sizeof(*arr));
    return arr;
}

float **points = malloc(Nx * sizeof(float *));


float **X_i = points[i][j][0];
float X_i1;
float a = 0.5, b= 0.5, dt = 0.01;



void normalize(float ** vector_array, int N) {
    // parallelize this section
    for (int i = 0 ; i < N; i++) {
        float magn = sqrt(vector_array[i][0]*vector_array[i][0] + vector_array[i][1]*vector_array[i][1]);  
        vector_array[i][0] = vector_array[i][0]/magn;
        vector_array[i][1] = vector_array[i][1]/magn;
    }
} 

float *get_inner_prod(float **X, float **Y, int N){
    float *inner_prod = get_new_array(N);
    // vectorize this section
    for (int i = 0 ; i < N; i++) {
        inner_prod[i] = X[i][0]*Y[i][0] + X[i][1]*Y[i][1];
    } 
    return inner_prod;
}


float* get_direct_flux(float x, float y) {

    float *direct_flux = get_new_array(1); 
    *direct_flux[0][0] = 0l
    *direct_flux[0][1] = 0;
    if (x > -0.5 && x < 0.5) {
        *direct_flux[0][1] = -1;
    }

    return direct_flux;
}


float* get_center(float** points, int N) {
    float *center = get_new_array(1);
    center[0][0] =  0;
    center[0][0] =  0;
    for  (int i = 0; i < N, i++) {
        center[0][0] = center[0][0] + points[i][0];
        center[0][1] = center[0][1] + points[i][1];
    }
    center[0][0] = center[0][0]/N;
    center[0][1] = center[0][1]/N;
    return center
}
float* get_indirect_flux(float x, float y, float** points, int N){
    float *indirect_flux = get_new_array(1); 
    float *center = get_center(float** points, int N);
}


float *get_etch_rate(t, float x_i, float y_i,) {
    float** flux = get_flux(x_i, y_i);
    float** normal = get_normal(points);
    float** normal_ves = get_inner_prod(points, flux);
}

// implementaiton of rk4 method
float* get_numerical_rate(float x_i, float y_i, int t, float dt, float *(*operation)(float, float, float)) {
    float* k1 = operation(t + dt, x_i, y_i);
    float* k2 = operation(t + dt/2, x_i + (*k1[0]/2)*dt, y_i + (*k1[1]/2)*dt);
    float* k3 = operation(t + dt/2, x_i + (*k2[0]/2)*dt, y_i + (*k2[1]/2)*dt);
    float* k4 = operation(t + dt, x_i + (*k3[0])*dt, (*k3[1])*dt);
    return k4;
}


void update_points(float ** points, float t, float dt) {
    for i in range(0, 100){
        float** flux = get_flux(points);
        float* etch_rate = get_numerical_rate(points[i][0], points[i][1], t, dt, );
        float *flux_projection = get_inner_prod();
         
    }
}

int main() {

    int N
    int Nx = 1e5;
    int Ny = 1e5;
    float **points = malloc(Nx * sizeof(float *));
    

    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel 
    {

    }
    for (int i = 0; i < Nx; i++) {
        points[i] = malloc(Ny * sizeof(int));
    }
    return 0;
}
