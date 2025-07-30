#pragma once


#include <cuda_runtime.h>
#include <map>
#include <vector>

#include "threads.h"


extern std::map<int, cudaStream_t> ThreadsStream;
extern std::vector<cudaEvent_t> Registered_Events;

extern cudaStream_t main_stream, backward_stream;