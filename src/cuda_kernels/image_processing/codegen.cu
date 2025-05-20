#pragma once


#include "../../codegen/random.h"
#include "../../common/cu_commons.h"
#include "../../compiler_frontend/logging.h"
#include "../../tensor/include.h"
#include "../calculate_grids.h"
#include "../handles.h"
#include "kernels.h"


extern "C" DT_tensor *RandomCrop(int thread_id, DT_tensor *tensor, float padding)
{
  float *tensor_ptr, *cropped;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<int> dims = tensor->dims;
  float dims_prod = tensor->dims_prod;

  unsigned long long seed = get_int_seed();

  float B, C, H, W;
  B = dims[0];
  C = dims[dims.size()-3];
  H = dims[dims.size()-2];
  W = dims[dims.size()-1];

  cropped = get_from_pool(thread_id, dims_prod, "cropping");


  int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

  dim3 numBlocks(B, C, std::ceil((H*W)/(float)block_size));
  dim3 threadsPerBlock(block_size);
  cudaCheck(cudaGetLastError());


  random_padding_cropping_kernel<<<numBlocks, threadsPerBlock, 0, tensor->cuda_stream>>>(
    tensor_ptr,
    cropped,
    B,
    C,
    H,
    W,
    H,
    W,
    padding,
    seed
  );
  cudaCheck(cudaGetLastError());
  
  DT_tensor *new_tensor = createTensor(cropped, dims, dims_prod, false, "", tensor->cuda_stream);
  new_tensor->AttrLNode(tensor, crop_op);
  return new_tensor;
}






extern "C" DT_tensor *RandomHorizontalFlip(int thread_id, DT_tensor *tensor)
{
  float *tensor_ptr, *flipped;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<int> dims = tensor->dims;
  float dims_prod = tensor->dims_prod;

  unsigned long long seed = get_int_seed();

  float B, C, H, W;
  B = dims[0];
  C = dims[dims.size()-3];
  H = dims[dims.size()-2];
  W = dims[dims.size()-1];

  flipped = get_from_pool(thread_id, dims_prod, "horizontal_flipping");


  int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

  dim3 numBlocks(B, C, std::ceil((H*W)/(float)block_size));
  dim3 threadsPerBlock(block_size);
  cudaCheck(cudaGetLastError());

  //std::cout << "B " << B << ", C " << C << ", H " << H << ", W " << W << "\n";

  random_horizontal_flip_kernel<<<numBlocks, threadsPerBlock, 0, tensor->cuda_stream>>>(
    tensor_ptr,
    flipped,
    B,
    C,
    H,
    W,
    seed
  );
  cudaCheck(cudaGetLastError());
  
  DT_tensor *new_tensor = createTensor(flipped, dims, dims_prod, false, "", tensor->cuda_stream);
  new_tensor->AttrLNode(tensor, random_horizontal_flip_op);
  return new_tensor;
}





extern "C" DT_tensor *NormalizeImg(int thread_id, DT_tensor *tensor, DT_tensor *mean, DT_tensor *std)
{
  float *tensor_ptr, *normalized;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<int> dims = tensor->dims;
  float dims_prod = tensor->dims_prod;


  float B, C, H, W;
  B = dims[0];
  C = dims[dims.size()-3];
  H = dims[dims.size()-2];
  W = dims[dims.size()-1];

  if(mean->dims_prod!=C||std->dims_prod!=C)
  { 
    LogErrorS("NormalizeImg mean and std tensors must have the same dimensionality as the image channels.");
    return nullptr;
  }

  normalized = get_from_pool(thread_id, dims_prod, "normalize img");



  int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

  dim3 numBlocks(B, C, std::ceil((H*W)/(float)block_size));
  dim3 threadsPerBlock(block_size);
  cudaCheck(cudaGetLastError());

  normalize_img_kernel<<<numBlocks, threadsPerBlock, 0, tensor->cuda_stream>>>(
    normalized,
    tensor_ptr,
    mean->tensor_ptr,
    std->tensor_ptr,
    B,
    C,
    H,
    W
  );

  cudaCheck(cudaGetLastError());

  DT_tensor *new_tensor = createTensor(normalized, dims, dims_prod, false, "", tensor->cuda_stream);
  new_tensor->AttrLNode(tensor, normalize_img_op);
  return new_tensor;
}


extern "C" DT_tensor *Jitter(int thread_id, DT_tensor *tensor, float factor)
{


  int grid_size, block_size;
  float dims_prod = tensor->dims_prod;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];


  float *jittered = get_from_pool(thread_id, dims_prod, "jitter img");
  unsigned long long seed = get_int_seed();
  jitter_kernel<<<grid_size, block_size, 0, tensor->cuda_stream>>>(jittered, tensor->tensor_ptr, factor, dims_prod, seed);


  DT_tensor *new_tensor = createTensor(jittered, tensor->dims, dims_prod, false, "", tensor->cuda_stream);
  new_tensor->AttrLNode(tensor, jitter_op);
  return new_tensor;
}