#pragma once


#include "threads.h"
#include <cuda_runtime.h>
#include <map>
#include <vector>



extern CudaStreams *main_stream, *backward_stream;
extern std::map<int, cudaStream_t> ThreadsStream;
extern std::vector<cudaEvent_t> Registered_Events;