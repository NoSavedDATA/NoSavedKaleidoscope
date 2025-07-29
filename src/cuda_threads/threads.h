#pragma once

#include <cuda_runtime.h>
#include <thread>
#include <vector>

struct CudaStreams {
  cudaStream_t stream;
  int idx;
};

CudaStreams *AllocateStream(int line=0);

cudaStream_t createCudaStream();


void SynchronizeStream(cudaStream_t cuda_stream);


struct Loader {
    std::vector<std::thread> threads;
    std::vector<CudaStreams *> streams;

    void Load(float *tensor_ptr, const float *tensor_cpu, int all_dims_prod);

    void Sync(); 
};


constexpr int num_parallel_streams=32; //global of src/mma/tensor_struc.h
constexpr int ASYNC_LOADER_THREADS = 6;

extern CudaStreams *parallel_streams[];
extern const int num_parallel_streams;
extern cudaEvent_t parallel_events[];
extern int open_streams[];


void RegisterEvent(cudaStream_t stream);

void WaitForAllEvents();


void StreamAwaitStreamB(cudaStream_t A, cudaStream_t B);
