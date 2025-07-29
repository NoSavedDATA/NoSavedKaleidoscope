#include <cublasLt.h>
#include <cublas_v2.h>
#include <cudnn.h>


cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
cudnnHandle_t cudnn;

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
size_t cublaslt_workspace_size = 32 * 1024 * 1024; // 32 MB
void* cublaslt_workspace = NULL;
cublasComputeType_t cublas_compute_type;


cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;


float eps = 1e-8;


cudaDeviceProp deviceProp;
int WARP_SIZE=32;
int THREADS_PER_BLOCK = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;