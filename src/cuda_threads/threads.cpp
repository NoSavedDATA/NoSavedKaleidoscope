#include <cmath>
#include <iostream>


#include "../common/extension_functions.h"
#include "../compiler_frontend/logging.h"
#include "include.h"




CudaStreams *parallel_streams[num_parallel_streams];
cudaEvent_t parallel_events[num_parallel_streams];
std::vector<cudaEvent_t> Registered_Events;
int open_streams[num_parallel_streams];


cudaStream_t main_stream, backward_stream;
std::map<int, cudaStream_t> ThreadsStream;


CudaStreams *AllocateStream(int line)
{
  int free_stream = FirstNonzero(open_streams, num_parallel_streams);
  if (free_stream<0)
    LogErrorCodegen("Failed to allocate a cuda stream. Probably loading too many different tensors.", line);
  open_streams[free_stream] = 0;
  //std::cout << "Allocating stream " << free_stream << "\n";
  return parallel_streams[free_stream];
}


cudaStream_t createCudaStream() {
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
      LogErrorS(-1, "Error allocating a cuda stream");
      std::exit(0);
    }
    return stream;
}


void SynchronizeStream(cudaStream_t cuda_stream)
{

  // Save below for later

  // cudaEvent_t memcpy_done_event;
  // cudaEventCreate(&memcpy_done_event);

  // cudaMemcpyAsync(tensor_ptr, tensor_cpu, batchless_dims_prod * sizeof(float),
  //                 cudaMemcpyHostToDevice, cuda_stream);
  // cudaEventRecord(memcpy_done_event, cuda_stream);

  // cudaStreamWaitEvent(main_stream, memcpy_done_event, 0);


  std::cout << "Synchronizing stream " << "\n";
  cudaStreamSynchronize(cuda_stream);
}




void Loader::Load(float *tensor_ptr, const float *tensor_cpu, int all_dims_prod) {

  // float quotient = std::floor(all_dims_prod / ASYNC_LOADER_THREADS);
  // float remainder = all_dims_prod % ASYNC_LOADER_THREADS;


  // std::vector<int> dims_prods;

  // for(int i=0; i<ASYNC_LOADER_THREADS-1; i++)
  //   dims_prods.push_back(quotient);
  // dims_prods.push_back(quotient+remainder);


  // float offset, size;
  // offset = 0;
  // for(int i=0; i<ASYNC_LOADER_THREADS; i++)
  // {
  //   size = dims_prods[i];
  //   CudaStreams *cuda_stream = AllocateStream(0);

  //   //copyChunk(tensor_ptr, tensor_cpu, offset, size, cuda_stream);
  //   //threads.push_back(std::thread(copyChunk, tensor_ptr, tensor_cpu, offset, size, cuda_stream));

  //   cudaMemcpyAsync(tensor_ptr + (int)offset, tensor_cpu + (int)offset, size*sizeof(float), cudaMemcpyHostToDevice, cuda_stream);

  //   streams.push_back(cuda_stream);
  //   offset += size;
  // }
}
    
void Loader::Sync()
{
  // for(int i=0; i<ASYNC_LOADER_THREADS; i++)
  // {
  //   SynchronizeStream(streams[i]);
  //   //threads[i].join();
  // }
  // streams.clear();
  // //threads.clear();
}


void RegisterEvent(cudaStream_t stream)
{
  //TODO: does this work inside threads?

  cudaEvent_t event;
  cudaEventCreate(&event);

  cudaEventRecord(event, stream);

  Registered_Events.push_back(event);
}
void WaitForAllEvents()
{
  while(Registered_Events.size()>0)
  {
    cudaEvent_t event = Registered_Events.back();
    Registered_Events.pop_back();

    cudaStreamWaitEvent(main_stream, event, 0);
    cudaEventDestroy(event);
  }
}


void StreamAwaitStreamB(cudaStream_t A, cudaStream_t B)
{
  // Create an event
  cudaEvent_t event;
  cudaEventCreate(&event);

  // Record the event when the kernel finishes execution on 'stream'
  cudaEventRecord(event, B);

  cudaStreamWaitEvent(A, event, 0);
  cudaEventDestroy(event);
}
