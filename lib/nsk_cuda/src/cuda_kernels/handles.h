#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cudnn.h>

extern cublasHandle_t cublas_handle;
extern cublasLtHandle_t cublaslt_handle;
extern cudnnHandle_t cudnn;

extern size_t cublaslt_workspace_size;
extern void* cublaslt_workspace;
extern cublasComputeType_t cublas_compute_type;

extern cublasComputeType_t cublas_compute;

extern const int TILE_SIZE; 
extern const int TILE_SIZE_SQ;

#define CUBLAS_LOWP CUDA_R_32F
#define PRECISION_MODE PRECISION_FP32

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)


extern float eps;

extern cudaDeviceProp deviceProp;


extern int WARP_SIZE;



extern int THREADS_PER_BLOCK;

extern const int TILE_SIZE; 
extern const int TILE_SIZE_SQ;

