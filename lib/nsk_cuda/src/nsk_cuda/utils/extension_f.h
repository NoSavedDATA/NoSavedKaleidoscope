#pragma once

#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

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



bool in_float_ptr_vec(const float *value, const std::vector<float *>& list);

bool in_half_ptr_vec(const half *value, const std::vector<half *>& list);

bool in_int8_ptr_vec(const int8_t* value, const std::vector<int8_t*>& list);