// REFERENCES:
// https://github.com/karpathy/llm.c/blob/master/dev/cuda/common.h

#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h>
#include <iostream>
#include <random>


#include <Eigen/Dense>
#include "cu_commons.h"

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

// cuBLAS error checking
void _cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}

// ----------------------------------------------------------------------------
// random utils

unsigned int get_millisecond_time() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<unsigned int>(ts.tv_sec + (ts.tv_nsec / 1000));
}

std::random_device rd;  // obtain a random seed
// std::mt19937 WEIGHT_PRNG(get_millisecond_time()); // initialize the Mersenne Twister generator
std::mt19937 WEIGHT_PRNG(rd()^get_millisecond_time()); // initialize the Mersenne Twister generator

//std::mt19937 WEIGHT_PRNG(0);

float* make_random_float_uniform(size_t N) {
    
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // range -1..1
    
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
        arr[i] = dist(WEIGHT_PRNG);
        //arr[i] = ((float)rand() / RAND_MAX); // range 0..1
    
    return arr;
}

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

float* make_random_int(size_t N, int V) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = (float)((int)(rand() % V)); // range 0..V-1
    }
    return arr;
}

float* make_arange(size_t N) {

    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
        arr[i] = i;
    
    return arr;
}

float* make_zeros_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}
float* make_ones_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    return arr;
}

float* make_min_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    float min_float = -50000;
    memset(arr, min_float, N * sizeof(float));
    return arr;
}

float* make_xavier_uniform_float(size_t N, int fan_in, int fan_out) {
    float xavier_scale = sqrt(6/(float)(fan_in+fan_out));

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
        arr[i] = xavier_scale*dist(WEIGHT_PRNG);
    
    return arr;
}


float* make_xavier_uniform_float_fixed(size_t N, int fan_in, int fan_out, int seed) {
    float xavier_scale = sqrt(6/(float)(fan_in+fan_out));

    std::mt19937 fixed_weight_prng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
        arr[i] = xavier_scale*dist(fixed_weight_prng);
    
    return arr;
}


float* make_normal(int N) {
    std::normal_distribution<float> dist(0.0, 1.0);

    float* arr = (float*)malloc(N * sizeof(float));
        
    for (int i = 0; i < N; i++)
        arr[i] = dist(WEIGHT_PRNG);
    
    return arr;
}

float* make_embedding_uniform(int N, float scale) {
    std::uniform_real_distribution<float> dist(-scale, scale);

    float* arr = (float*)malloc(N * sizeof(float));
        
    for (int i = 0; i < N; i++)
        arr[i] = dist(WEIGHT_PRNG);
    
    return arr;
}


float *make_orthogonal(size_t rows, size_t cols) {
    // Generate a random matrix using normal distribution
    //float scale = sqrt(1/rows);
    
    std::normal_distribution<float> dist(0.0f, 1.0f);

    Eigen::MatrixXf random_matrix(rows, cols);
    
    // Fill the matrix with random values
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            random_matrix(i, j) = dist(WEIGHT_PRNG);
        

    // Perform QR decomposition to get orthogonal matrix Q
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(random_matrix);
    Eigen::MatrixXf Q = qr.householderQ();


    if (rows >= cols)
        Q = Q.leftCols(cols); // First "cols" columns are orthogonal
    else
        Q = Q.topRows(rows); // First "rows" rows are orthogonal


    // Allocate memory for the orthogonal weights
    float* weights = (float*)malloc(rows * cols * sizeof(float));

    // Copy the orthogonal matrix Q to the weight array
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            weights[i * cols + j] = Q(i, j);
            //weights[i * cols + j] = scale*Q(i, j);
    

    return weights;
}


void make_1_orthogonal(int n, float *w, size_t rows, size_t cols) {
    // Generate a random matrix using normal distribution

    //float scale = sqrt(2/(rows+cols));
    //float scale = sqrt(1/(rows));
    
    std::normal_distribution<float> dist(0.0f, 1.0f);

    Eigen::MatrixXf random_matrix(rows, cols);
    
    // Fill the matrix with random values
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            random_matrix(i, j) = dist(WEIGHT_PRNG);

    

    // Perform QR decomposition to get orthogonal matrix Q
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(random_matrix);
    Eigen::MatrixXf Q = qr.householderQ();

    
    if (rows >= cols)
        Q = Q.leftCols(cols); // First "cols" columns are orthogonal
    else
        Q = Q.topRows(rows); // First "rows" rows are orthogonal

    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            w[n*rows*cols + i * cols + j] = Q(i, j);
    

}

float *make_N_orthogonals(int N, size_t rows, size_t cols)
{
    float* weights = (float*)malloc(N * rows * cols * sizeof(float));

    for (int i=0; i<N; i++)
        make_1_orthogonal(i, weights, rows, cols);

    return weights;
}


float* make_xavier_uniform_float_relu(size_t N, int fan_in, int fan_out) {
    float xavier_scale = 1.4142*sqrt(6/(float)(fan_in+fan_out));

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f); // range -1..1

    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
        arr[i] = xavier_scale*dist(WEIGHT_PRNG);
        //arr[i] = xavier_scale*(((float)rand() / RAND_MAX) * 2.0 - 1.0); // range -1..1
    
    return arr;
}

float* make_xavier_uniform_float_tanh(size_t N, int fan_in, int fan_out) {
    float xavier_scale = 1.6667*sqrt(6/(float)(fan_in+fan_out));

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f); // range -1..1

    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
        arr[i] = xavier_scale*dist(WEIGHT_PRNG);
        //arr[i] = xavier_scale*(((float)rand() / RAND_MAX) * 2.0 - 1.0); // range -1..1
    
    return arr;
}



float* make_he_normal_float_relu(int N, int fan_in) {
    float std = sqrt(2/(float)fan_in);

    std::normal_distribution<> dist(0.0, std);

    float* arr = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++)
        arr[i] = dist(WEIGHT_PRNG);

    return arr;
}

float* make_gpt_init(int N) {
    
    std::normal_distribution<double> dist(0.0, 0.02);

    float* arr = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++)
        arr[i] = dist(WEIGHT_PRNG);
    
    return arr;
}

float* make_lstm_init_xavier(int OC, int C) {

    float xavier_scale = sqrt(6/(OC+C));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f); // range -1..1
    
    //float xavier_scale = sqrt(2/(OC+C));
    //std::normal_distribution<float> dist(0.0, xavier_scale);
    

    float* arr = (float*)malloc(4*OC*C * sizeof(float));

    int tanh_offset = 3*(int)OC*(int)C;
    

    //for (int i = 0; i < 4*OC*C; i++)
    //    arr[i] = xavier_scale*dist(WEIGHT_PRNG);
    
    for (int i = 0; i < tanh_offset; i++)
        arr[i] = xavier_scale*dist(WEIGHT_PRNG);
    for (int i = 0; i < OC*C; i++)
        arr[i+tanh_offset] = 1.667*xavier_scale*dist(WEIGHT_PRNG);
    
    

    
    return arr;
}

float* make_lstm_bias(int OC) {


    float* arr = (float*)malloc(4*OC * sizeof(float));

    int f_offset = 1*OC;
    int o_offset = 2*OC;
    

    for (int i = 0; i < f_offset; i++)
        arr[i] = 0.0f;
    
    for (int i = 0; i < OC; i++)
        arr[f_offset+i] = 1.0f;

    for (int i = 0; i < OC*2; i++)
        arr[o_offset+i] = 0.0f;
    

    
    return arr;
}


float* make_lstm_torch(int OC, int C) {

    float scale = sqrt(1/(float)(4*OC));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f); // range -1..1

    float* arr = (float*)malloc(4*OC*C * sizeof(float));

    for (int i = 0; i < 4*OC*C; i++)
        arr[i] = scale*dist(WEIGHT_PRNG);
    
    return arr;
}

