#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include <math.h>

linear etch rate

float **X_i = points[i][j][0];
float X_i1;
float a = 0.5, b= 0.5, dt = 0.01;

float rate(float X, float Y) {
    return exp(-(X**2))
}

float project_x(float **X, float **Y, int i){
    X[i]
}

float update_position(float **X_i, float **Y_i, int N, float (*operation)(float, float)) {
    for (int i = 0; i < N; i++){
        float X_i1 = (a*rate(X_i, Y_i) + b*rate(X_i1, Y_i1))*dt + X_i;
    }
    
}



struct Vector2D() {
    int ** vector;  
}

float calculte_rate_project(float rate, float **points, int i, int j){
    normal_vect = calculate_normal()
}

int main() {

    int N
    int Nx = 1e5;
    int Ny = 1e5;
    float **points = malloc(Nx * sizeof(float *));
    
    #pragma omp parallel for
    for (int i = 0; i < Nx; i++) {
        points[i] = malloc(Ny * sizeof(int));
    }
    return 0;
}