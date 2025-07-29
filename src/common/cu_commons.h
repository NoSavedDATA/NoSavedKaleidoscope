#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>

#include <iostream>
#include <cudnn.h>
#include <cstdlib>

// REFERENCES:
// https://github.com/karpathy/llm.c/blob/master/dev/cuda/common.h



template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
void cuda_check(cudaError_t error, const char *file, int line);
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
void _cublasCheck(cublasStatus_t status, const char *file, int line);

#define cublasCheck(status) (_cublasCheck((status), __FILE__, __LINE__))


#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

// ----------------------------------------------------------------------------
// random utils


extern std::random_device rd;  // obtain a random seed
extern std::mt19937 WEIGHT_PRNG; // initialize the Mersenne Twister generator
//std::mt19937 WEIGHT_PRNG(0);

float* make_random_float_uniform(size_t N); 

float* make_random_float(size_t N);;

float* make_random_int(size_t N, int V); 

float* make_arange(size_t N); 

float* make_zeros_float(size_t N); 
float* make_ones_float(size_t N); 

float* make_min_float(size_t N); 

float* make_xavier_uniform_float(size_t N, int fan_in, int fan_out); 

float* make_xavier_uniform_float_fixed(size_t N, int fan_in, int fan_out, int seed);


float* make_normal(int N); 

float* make_embedding_uniform(int N, float scale=0.05); 


float *make_orthogonal(size_t rows, size_t cols); 


void make_1_orthogonal(int n, float *w, size_t rows, size_t cols); 

float *make_N_orthogonals(int N, size_t rows, size_t cols);



float* make_xavier_uniform_float_relu(size_t N, int fan_in, int fan_out); 

float* make_xavier_uniform_float_tanh(size_t N, int fan_in, int fan_out); 



float* make_he_normal_float_relu(int N, int fan_in); 

float* make_gpt_init(int N); 

float* make_lstm_init_xavier(int OC, int C); 

float* make_lstm_bias(int OC); 


float* make_lstm_torch(int OC, int C); 

