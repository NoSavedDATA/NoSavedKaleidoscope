#pragma once

#include "../../backprop/include.h"
#include "../../mangler/scope_struct.h"
#include "../../tensor/include.h"
#include "../activation_functions/include.h"
#include "../calculate_grids.h"
#include "../elementwise_kernels_inline.cu"
#include "classification_kernels.h"


void CrossEntropyBackward(Tensor *L_tensor, Tensor *R_tensor,
                          float *dloss,
                          float scale)
{
  
  /*
  int grid_size = B;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  */

  float *y_hat = L_tensor->tensor_ptr;
  float *y = R_tensor->tensor_ptr;
  std::vector<float> BC = format_LinearLayer_Dims(L_tensor->dims);
  float B  = BC[0];
  float C  = BC[1];
  

  float *probs = get_from_pool(0, B*C,"ce probs");

  //int grid_size, block_size;
  //size_t shared_mem_size;
  

  int grid_size, block_size, shared_mem_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size  = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  set_to_zero_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(probs, B*C);


  /*
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*32*C);
  grid_size  = B*32;
  block_size = grid_block_mem_sizes[1];
  
  online_softmax<<<grid_size, block_size, 0, main_stream->stream>>>(y_hat, probs, B, C);
  */
  
  
  

  
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size  = B;
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = 2 * block_size / 32 * sizeof(float);

  softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size, main_stream->stream>>>(y_hat, probs, B, C);
  
  


  
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  
  crossentropy_softmax_backward_kernel1<<<grid_size, block_size, 0, main_stream->stream>>>(dloss, probs, y, B, C, scale);
  move_to_pool(0, B*C, probs,"ce probs");

  
}



extern "C" float cross_entropy(Scope_Struct *scope_struct, Tensor *y_hat, Tensor *y, float scale)
{
  
  Tensor *loss_tensor = new Tensor();


  loss_tensor->AttrNodes(y_hat, y, cross_entropy_op);
  loss_tensor->scalar = scale;


  todo_backward_tensors.push_back(loss_tensor);

  

  return 0;
}


void CrossEntropyIdxBackward(Tensor *L_tensor, Tensor *R_tensor, 
                          float *dloss,
                          float scale)
{
  float *y_hat = L_tensor->tensor_ptr;
  float *y = R_tensor->tensor_ptr; 
  std::vector<float> BC = format_LinearLayer_Dims(L_tensor->dims);
  float B  = BC[0];
  float C  = BC[1];
  
  float *probs = get_from_pool(0, B*C,"ce probs");

  int grid_size, block_size, shared_mem_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size  = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  set_to_zero_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(probs, B*C);


  

  /*
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size  = B;
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = 2 * block_size / 32 * sizeof(float);

  softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size, main_stream->stream>>>(y_hat, probs, B, C);
  */
  grid_block_mem_sizes = CalculateSimpleWarpGridAndBlockSizes(B);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  online_softmax<<<grid_size, block_size, 0, main_stream->stream>>>(y_hat, probs, B, C);
  
  


  
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  
  crossentropy_idx_backward_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(dloss, probs, y, B, C, scale);
  move_to_pool(0, B*C, probs,"ce probs");
}



extern "C" float cross_entropy_idx(Scope_Struct *scope_struct, Tensor *y_hat, Tensor *y, float scale)
{
  
  Tensor *loss_tensor = new Tensor();


  loss_tensor->AttrNodes(y_hat, y, cross_entropy_idx_op);
  loss_tensor->scalar = scale;


  todo_backward_tensors.push_back(loss_tensor);

  

  return 0;
}