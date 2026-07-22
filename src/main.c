#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define NUM_THREADS 12

// Global variables for RK4 normal lookup
float **global_points = NULL;
float **global_normals = NULL;
int global_N = 0;

// Helper to allocate memory for N points (2D coordinates) in a contiguous block
float **get_new_points(int N) {
    float **arr = malloc(N * sizeof(float *));
    float *data = calloc(N * 2, sizeof(float));
    for (int i = 0; i < N; i++) {
        arr[i] = &data[i * 2];
    }
    return arr;
}

// Helper to free memory allocated by get_new_points
void free_points(float **arr) {
    if (arr) {
        free(arr[0]);
        free(arr);
    }
}

// Helper to allocate an array of N floats
float *get_new_array(int N) {
    return malloc(N * sizeof(float));
}

// Normalize function implementing the parallelization/vectorization strategy
void normalize(float **vector_array, int N) {
    if (N >= 1000000) {
        #pragma omp parallel for simd schedule(static)
        for (int i = 0; i < N; i++) {
            float magn = sqrtf(vector_array[i][0]*vector_array[i][0] + vector_array[i][1]*vector_array[i][1]);
            if (magn > 1e-6f) {
                vector_array[i][0] /= magn;
                vector_array[i][1] /= magn;
            }
        }
    } else if (N >= 100) {
        #pragma omp simd
        for (int i = 0; i < N; i++) {
            float magn = sqrtf(vector_array[i][0]*vector_array[i][0] + vector_array[i][1]*vector_array[i][1]);
            if (magn > 1e-6f) {
                vector_array[i][0] /= magn;
                vector_array[i][1] /= magn;
            }
        }
    } else {
        for (int i = 0; i < N; i++) {
            float magn = sqrtf(vector_array[i][0]*vector_array[i][0] + vector_array[i][1]*vector_array[i][1]);
            if (magn > 1e-6f) {
                vector_array[i][0] /= magn;
                vector_array[i][1] /= magn;
            }
        }
    }
}

// Inner product calculation implementing the parallelization/vectorization strategy
float *get_inner_prod(float **X, float **Y, int N) {
    float *inner_prod = get_new_array(N);
    if (N >= 1000000) {
        #pragma omp parallel for simd schedule(static)
        for (int i = 0; i < N; i++) {
            inner_prod[i] = X[i][0]*Y[i][0] + X[i][1]*Y[i][1];
        }
    } else if (N >= 100) {
        #pragma omp simd
        for (int i = 0; i < N; i++) {
            inner_prod[i] = X[i][0]*Y[i][0] + X[i][1]*Y[i][1];
        }
    } else {
        for (int i = 0; i < N; i++) {
            inner_prod[i] = X[i][0]*Y[i][0] + X[i][1]*Y[i][1];
        }
    }
    return inner_prod;
}

// Get direct etching flux
float *get_direct_flux(float x, float y) {
    float *direct_flux = get_new_array(2);
    direct_flux[0] = 0.0f;
    direct_flux[1] = 0.0f;
    if (fabsf(x) < 0.5f || y > 0.1f) {
        direct_flux[1] = 1.0f;
    }
    return direct_flux;
}

// Get center of points array implementing the parallelization/vectorization strategy
float *get_center(float **points, int N) {
    float *center = get_new_array(2);
    double sum_x = 0;
    double sum_y = 0;
    if (N >= 1000000) {
        #pragma omp parallel for reduction(+:sum_x, sum_y) schedule(static)
        for (int i = 0; i < N; i++) {
            sum_x += points[i][0];
            sum_y += points[i][1];
        }
    } else {
        for (int i = 0; i < N; i++) {
            sum_x += points[i][0];
            sum_y += points[i][1];
        }
    }
    center[0] = (float)(sum_x / N);
    center[1] = (float)(sum_y / N);
    return center;
}

// Get indirect flux
float *get_indirect_flux(float x, float y, float **points, int N) {
    float *indirect_flux = get_new_array(2);
    float *center = get_center(points, N);
    float dx = x - center[0];
    float dy = y - center[1];
    float dist = sqrtf(dx*dx + dy*dy);
    if (dist > 1e-6f) {
        indirect_flux[0] = dx / dist;
        indirect_flux[1] = dy / dist;
    } else {
        indirect_flux[0] = 0.0f;
        indirect_flux[1] = 0.0f;
    }
    free(center);
    return indirect_flux;
}

// Calculate normal vectors for points
float **get_normal(float **points, int N) {
    float **norm = get_new_points(N);
    norm[0][0] = 0.0f;
    norm[0][1] = 1.0f;
    norm[N-1][0] = 0.0f;
    norm[N-1][1] = 1.0f;

    if (N >= 1000000) {
        #pragma omp parallel for simd schedule(static)
        for (int i = 1; i < N - 1; i++) {
            float dx = points[i+1][0] - points[i-1][0];
            float dy = points[i+1][1] - points[i-1][1];
            float dd = sqrtf(dx*dx + dy*dy);
            if (dd > 1e-6f) {
                float theta = asinf(dy/dd) + 1.57079632679f;
                norm[i][0] = cosf(theta);
                norm[i][1] = sinf(theta);
            } else {
                norm[i][0] = 0.0f;
                norm[i][1] = 1.0f;
            }
        }
    } else if (N >= 100) {
        #pragma omp simd
        for (int i = 1; i < N - 1; i++) {
            float dx = points[i+1][0] - points[i-1][0];
            float dy = points[i+1][1] - points[i-1][1];
            float dd = sqrtf(dx*dx + dy*dy);
            if (dd > 1e-6f) {
                float theta = asinf(dy/dd) + 1.57079632679f;
                norm[i][0] = cosf(theta);
                norm[i][1] = sinf(theta);
            } else {
                norm[i][0] = 0.0f;
                norm[i][1] = 1.0f;
            }
        }
    } else {
        for (int i = 1; i < N - 1; i++) {
            float dx = points[i+1][0] - points[i-1][0];
            float dy = points[i+1][1] - points[i-1][1];
            float dd = sqrtf(dx*dx + dy*dy);
            if (dd > 1e-6f) {
                float theta = asinf(dy/dd) + 1.57079632679f;
                norm[i][0] = cosf(theta);
                norm[i][1] = sinf(theta);
            } else {
                norm[i][0] = 0.0f;
                norm[i][1] = 1.0f;
            }
        }
    }
    return norm;
}

// Get etch rate at a specific coordinate
float *get_etch_rate_op(float t, float x, float y) {
    float *flux = get_direct_flux(x, y);
    float flux_mag = sqrtf(flux[0]*flux[0] + flux[1]*flux[1]);

    float nx = 0.0f;
    float ny = 1.0f;
    float min_dist_sq = 1e30f;
    int best_idx = 0;

    for (int i = 0; i < global_N; i++) {
        float dx = x - global_points[i][0];
        float dy = y - global_points[i][1];
        float dist_sq = dx*dx + dy*dy;
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            best_idx = i;
        }
    }

    if (global_normals) {
        nx = global_normals[best_idx][0];
        ny = global_normals[best_idx][1];
    }

    if (best_idx == 0 || best_idx == 1 || best_idx == global_N - 2 || best_idx == global_N - 1) {
        nx = 0.0f;
        ny = 0.0f;
    }

    float *rate = get_new_array(2);
    rate[0] = flux_mag * nx;
    rate[1] = flux_mag * ny;

    free(flux);
    return rate;
}

// RK4 numerical rate calculator
float *get_numerical_rate(float x_i, float y_i, float t, float dt, float *(*operation)(float, float, float)) {
    float *k1 = operation(t, x_i, y_i);
    float *k2 = operation(t + dt/2.0f, x_i + (k1[0]/2.0f)*dt, y_i + (k1[1]/2.0f)*dt);
    float *k3 = operation(t + dt/2.0f, x_i + (k2[0]/2.0f)*dt, y_i + (k2[1]/2.0f)*dt);
    float *k4 = operation(t + dt, x_i + k3[0]*dt, y_i + k3[1]*dt);

    float *rate = get_new_array(2);
    rate[0] = (k1[0] + 2.0f*k2[0] + 2.0f*k3[0] + k4[0]) / 6.0f;
    rate[1] = (k1[1] + 2.0f*k2[1] + 2.0f*k3[1] + k4[1]) / 6.0f;

    free(k1);
    free(k2);
    free(k3);
    free(k4);
    return rate;
}

// Update coordinates using the RK4 method
void update_points(float **points, int N, float t, float dt) {
    global_points = points;
    global_N = N;
    global_normals = get_normal(points, N);

    float **new_pts = get_new_points(N);

    new_pts[0][0] = points[0][0];
    new_pts[0][1] = points[0][1];
    new_pts[1][0] = points[1][0];
    new_pts[1][1] = points[1][1];

    new_pts[N-2][0] = points[N-2][0];
    new_pts[N-2][1] = points[N-2][1];
    new_pts[N-1][0] = points[N-1][0];
    new_pts[N-1][1] = points[N-1][1];

    if (N >= 1000) {
        printf("\n ======================================\n");
        printf("running parallel code");
        printf("\n ======================================\n");
        #pragma omp parallel for schedule(dynamic)
        for (int i = 2; i < N - 2 ; i++) {
            float *rate = get_numerical_rate(points[i][0], points[i][1], t, dt, get_etch_rate_op);
            new_pts[i][0] = points[i][0] + rate[0] * dt;
            new_pts[i][1] = points[i][1] + rate[1] * dt;
            free(rate);
            if (i%1000 == 0){
                // printf("\n ======================================\n");
                printf("iter %i ", i);
                // printf("\n ======================================\n");
            }
        }
    } else {
        printf("running sequential code");
        for (int i = 0; i < N; i++) {
            float *rate = get_numerical_rate(points[i][0], points[i][1], t, dt, get_etch_rate_op);
            new_pts[i][0] = points[i][0] + rate[0] * dt;
            new_pts[i][1] = points[i][1] + rate[1] * dt;
            free(rate);
        }
    }

    for (int i = 0; i < N; i++) {
        points[i][0] = new_pts[i][0];
        points[i][1] = new_pts[i][1];
    }

    free_points(new_pts);
    free_points(global_normals);
    global_normals = NULL;
}

// Initialise boundary/plane coordinates
float **get_plane_points(int N, float a) {
    float **pts = get_new_points(N);
    pts[0][0] = -1.0f;
    pts[0][1] = a;
    pts[N-1][0] = 1.0f;
    pts[N-1][1] = a;

    pts[1][0] = -0.5f - 0.1f;
    pts[1][1] = a;
    pts[N-2][0] = 0.5f + 0.1f;
    pts[N-2][1] = a;

    if (N >= 100000)
    {
        # pragma omp parallel for
        for (int i = 2; i < N - 2; i++) {
            pts[i][0] = (float)(i - N/2.0f)/N;
            pts[i][1] = a;
        }
    } else {
        for (int i = 2; i < N - 2; i++) {
            pts[i][0] = (float)(i - N/2.0f)/N;
            pts[i][1] = a;
        }
    }
    
    return pts;
}

void run_simulation(float *points_in_out, float *trajectory_out, int N, int Nt, float dt) {
    omp_set_num_threads(NUM_THREADS);
    float **points = malloc(N * sizeof(float *));
    for (int i = 0; i < N; i++) {
        points[i] = &points_in_out[i * 2];
    }
    
    for (int step = 0; step < Nt; step++) {
        update_points(points, N, step * dt, dt);
        if (trajectory_out) {
            for (int i = 0; i < N; i++) {
                trajectory_out[step * N * 2 + i * 2] = points[i][0];
                trajectory_out[step * N * 2 + i * 2 + 1] = points[i][1];
            }
        }
    }
    
    free(points);
}

#ifndef SHARED_LIB
int main() {
    int N = 10000;
    float a = 0.1f;
    float dt = 0.01f;

    omp_set_num_threads(NUM_THREADS);

    float **points = get_plane_points(N, a);

    printf("Starting etching simulation with N = %d points...\n", N);
    printf("Initial point[N/2]: (%f, %f)\n", points[N/2][0], points[N/2][1]);

    for (int step = 0; step < 10; step++) {
        update_points(points, N, step * dt, dt);
    }

    printf("Simulation completed successfully.\n");
    printf("Final point[N/2]: (%f, %f)\n", points[N/2][0], points[N/2][1]);

    free_points(points);
    return 0;
}
#endif
