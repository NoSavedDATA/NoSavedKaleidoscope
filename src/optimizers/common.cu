#include "../backprop/include.h"
#include "../cuda_kernels/calculate_grids.h"
#include "../cuda_kernels/elementwise_kernels_inline.cu"
#include "../cuda_threads/include.h"

#include "interface.h"



std::unique_ptr<Optimizer> optimize(std::unique_ptr<Optimizer> optimizer)
{
  int num_streams = NamedParamGrads.size();

  std::vector<cudaStream_t> streams(num_streams);

  for (int i = 0; i < num_streams; ++i)
  {

    cudaStreamCreate(&streams[i]);
    //StreamAwaitStreamB(streams[i], main_stream->stream);
  }

  cudaStreamSynchronize(main_stream->stream);

  int i=0;
  for (auto& pair : NamedParamGrads)
  {
    std::string param_name = pair.first;
    //std::cout << "Optimizing " << param_name << "\n";

    if (param_name!="none")
    {
      float *grad = pair.second;
      Tensor *tensor = NamedTensorsT[param_name];
      
      //std::cout << "param dims: "  << "\n";
      //PrintDims(tensor->dims);
      optimizer->init_states(param_name, tensor->dims_prod);

      if (tensor->Sparse_Idx_Tensor!=nullptr)
      {
        //std::cout << "Tensor " << param_name << " has a sparse gradient "<< "\n";
        Tensor *idx_tensor = tensor->Sparse_Idx_Tensor;

        optimizer->sparse_step(tensor->tensor_ptr, grad, idx_tensor->tensor_ptr,
                               idx_tensor->dims, tensor->dims, param_name, streams[i]);

        move_to_pool(0, idx_tensor->dims_prod, idx_tensor->tensor_ptr, "sparse grad idxs");
        delete idx_tensor;
      } else
        optimizer->step(tensor->tensor_ptr, grad, tensor->dims, param_name, streams[i]);

      int grid_size, block_size; 
      std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(tensor->dims_prod);
      grid_size = grid_block_mem_sizes[0];
      block_size = grid_block_mem_sizes[1];

      set_to_zero_kernel<<<grid_size, block_size, 0, streams[i]>>>(grad, tensor->dims_prod);
    }
    i+=1;
  }
  optimizer->count_step();

  
  for (int i = 0; i < num_streams; ++i)
  {
    cudaStreamSynchronize(streams[i]);
    //StreamAwaitStreamB(main_stream->stream, streams[i]);
  }
  for (int i = 0; i < num_streams; ++i)
    cudaStreamDestroy(streams[i]);

  cudaStreamSynchronize(main_stream->stream);

  return std::move(optimizer);
}


std::unique_ptr<Optimizer> optimizer = nullptr;

