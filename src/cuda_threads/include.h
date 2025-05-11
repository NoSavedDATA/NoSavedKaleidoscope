#pragma once


#include "threads.h"
#include <cuda_runtime.h>
#include <map>
#include <vector>



extern std::map<int, cudaStream_t> ThreadsStream;
extern std::vector<cudaEvent_t> Registered_Events;

extern cudaStream_t main_stream, backward_stream;