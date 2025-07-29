#pragma once

#include <iostream>
#include <vector>

#include "../common/extension_functions.h"
#include "../compiler_frontend/logging.h"
#include "../cuda_kernels/calculate_grids.h"
#include "../cuda_kernels/elementwise_kernels_inline.cu"
#include "../data_types/include.h"
#include "../tensor/tensor_struct.h"
#include "networks.h"


extern "C" float network_ema(int thread_id, char *scope, char *_ema_network, char *_network, float factor)
{

  std::string ema_network, network;
  ema_network = NamedObjects[_ema_network];
  network = NamedObjects[_network];

  DT_tensor *ema_tensor, *net_tensor;
  cudaStream_t stream = ThreadsStream[thread_id];


  //std::cout << "\nNETWORK EMA OF " << ema_network << " and " << network << "\n\n";

  //std::cout << "\n\n\n\n\n\n\n\n\n\nNETWORK EMA OF " << ema_network << " and " << network << "\n";



  std::vector<std::string> ema_params, net_params;
  for (const auto &pair : NamedTensorsT)
  {
    std::string param_name = pair.first;
    if (starts_with(param_name.c_str(), ema_network.c_str()))
      ema_params.push_back(param_name);
    
    if (starts_with(param_name.c_str(), network.c_str()))
      net_params.push_back(param_name);
    
  }


  for (const auto &ema_param : ema_params)
  {
    for (const auto &net_param : net_params)
    {
      if (contains_str(net_param, remove_substring(ema_param, ema_network)))
      {
        //std::cout << "MATCHED PARAMETER: " << ema_param  << " and " << net_param << "\n";

        ema_tensor = NamedTensorsT[ema_param];
        net_tensor = NamedTensorsT[net_param];

        if (ema_tensor->dims_prod!=net_tensor->dims_prod)
        {
          std::string _error = "network_ema failed because " + ema_tensor->name + " and " + net_tensor->name + " parameter sizes do not match.";
          LogErrorS(-1, _error);
        } else {

          int grid_size, block_size;
          std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(ema_tensor->dims_prod);
          grid_size = grid_block_mem_sizes[0];
          block_size = grid_block_mem_sizes[1];

          net_tensor->Sync();
          ema_tensor->Sync();

          
          ema_tensor_kernel<<<grid_size, block_size, 0, stream>>>(ema_tensor->tensor_ptr, net_tensor->tensor_ptr, factor, ema_tensor->dims_prod);

        }
      }
    }
  }

  cudaStreamSynchronize(stream);

  //starts_with

  //std::cout  << "\n\n\n\n\n\n\n\n\n\n";

  return 0;
}