
#include<cstdarg>
#include<vector>
#include<map>
#include<iostream>
#include<random>
#include<thread>
#include<string>

#include "../codegen/string.h"
#include "../common/include.h"
#include "../compiler_frontend/logging.h"
#include "../cuda_kernels/calculate_grids.h"
#include "../cuda_kernels/dim_kernels.h"
#include "../data_types/include.h"
#include "../tensor/include.h"
#include "include.h"




extern "C" void PrintDims(std::vector<int> dims)
{
  std::cout << "dims: [";
  for (int i=0; i<dims.size();i++)
  {
    std::cout << dims[i];
    if (i==dims.size()-1)
      std::cout << "]";
    else
      std::cout << ", ";
  }
  std::cout  << "\n";
}

int DimsProd(std::vector<float> dims)
{
  if (dims.size()==1)
    return (int) dims[0];

  float aux=1;
  for (int i = 0; i < dims.size(); i++)
    aux = aux*dims[i];
  return (int)aux;
}

int DimsProd(std::vector<int> dims)
{
  if (dims.size()==1)
    return dims[0];

  int aux=1;
  for (int i = 0; i < dims.size(); i++)
    aux = aux*dims[i];
  return aux;
}

std::vector<float> BatchLessDims(std::vector<float> dims)
{
  // Removes first dim (batch dim).
  if (dims.size()<=1)
    LogError("Cannot remove the batch dimension of a unidimensional tensor.");

  std::vector<float> new_dims;

  for (int i=0; i<dims.size()-1;i++)
    new_dims.push_back(dims[i+1]);

  return new_dims;
}

std::vector<int> BatchLessDims(std::vector<int> dims)
{
  // Removes first dim (batch dim).
  if (dims.size()<=1)
    LogError("Cannot remove the batch dimension of a unidimensional tensor.");

  std::vector<int> new_dims;

  for (int i=0; i<dims.size()-1;i++)
    new_dims.push_back(dims[i+1]);

  return new_dims;
}


std::vector<float> RemoveLastDim(std::vector<float> dims)
{
  // Removes first dim (batch dim).
  if (dims.size()<=1)
  {
    return {1.0f};
    //LogError("Cannot remove the batch dimension of a unidimensional tensor.");
  }

  std::vector<float> new_dims;

  for (int i=0; i<dims.size()-1;i++)
    new_dims.push_back(dims[i]);

  return new_dims;
}

std::vector<float> RemoveFirstDim(std::vector<float> dims)
{
  // Removes first dim (batch dim).
  if (dims.size()<=1)
    return {1.0f};

  std::vector<float> new_dims;

  for (int i=0; i<dims.size()-1;i++)
    new_dims.push_back(dims[i+1]);

  return new_dims;
}


std::vector<int> RemoveLastDim(std::vector<int> dims)
{
  // Removes first dim (batch dim).
  if (dims.size()<=1)
  {
    return {1};
    //LogError("Cannot remove the batch dimension of a unidimensional tensor.");
  }

  std::vector<int> new_dims;

  for (int i=0; i<dims.size()-1;i++)
    new_dims.push_back(dims[i]);

  return new_dims;
}

std::vector<int> RemoveFirstDim(std::vector<int> dims)
{
  // Removes first dim (batch dim).
  if (dims.size()<=1)
    return {1};

  std::vector<int> new_dims;

  for (int i=0; i<dims.size()-1;i++)
    new_dims.push_back(dims[i+1]);

  return new_dims;
}


std::vector<float> format_BatchFirst_Dims(std::vector<float> dims)
{
  std::vector<float> new_dims;
  new_dims.push_back(dims[0]);
  int aux=1;
  for (int i = 0; i < dims.size()-1; i++)
    aux *= dims[i+1];
  new_dims.push_back(aux);
  return new_dims;
}

std::vector<int> format_BatchFirst_Dims(std::vector<int> dims)
{
  std::vector<int> new_dims;
  new_dims.push_back(dims[0]);
  int aux=1;
  for (int i = 0; i < dims.size()-1; i++)
    aux *= dims[i+1];
  new_dims.push_back(aux);
  return new_dims;
}


std::vector<float> format_LinearLayer_Dims(std::vector<float> dims)
{
  std::vector<float> new_dims;
  int aux=1;
  for (int i = 0; i < dims.size()-1; i++)
    aux *= dims[i];
  new_dims.push_back(aux);
  new_dims.push_back(dims[dims.size()-1]);
  return new_dims;
}


std::vector<int> format_LinearLayer_Dims(std::vector<int> dims)
{
  std::vector<int> new_dims;
  int aux=1;
  for (int i = 0; i < dims.size()-1; i++)
    aux *= dims[i];
  new_dims.push_back(aux);
  new_dims.push_back(dims[dims.size()-1]);
  return new_dims;
}



int resultingDimsProdOnMult(std::vector<float> Ldims, std::vector<float> Rdims)
{
  float aux=1;
  for (int i = 0; i < Ldims.size()-1; i++)
    aux = aux * Ldims[i];
  aux = aux * Rdims[0];
  return (int)aux;
}

int resultingDimsProdOnMult(std::vector<int> Ldims, std::vector<int> Rdims)
{
  float aux=1;
  for (int i = 0; i < Ldims.size()-1; i++)
    aux = aux * Ldims[i];
  aux = aux * Rdims[0];
  return aux;
}



std::vector<float> NewDimsOnMult(std::vector<float> Ldims, std::vector<float> Rdims)
{
  

  std::vector<float> new_dims;
  if (Ldims[Ldims.size()-1]!=Rdims[Rdims.size()-1])
  {
    LogError("The last dimension of multiplied tensors must be the same.");
    return {}; 
  }
  for (int i = 0; i < Ldims.size()-1; i++)
    new_dims.push_back(Ldims[i]);
  new_dims.push_back(Rdims[0]);


  return new_dims;
}


std::vector<int> NewDimsOnMult(std::vector<int> Ldims, std::vector<int> Rdims)
{
  

  std::vector<int> new_dims;
  if (Ldims[Ldims.size()-1]!=Rdims[Rdims.size()-1])
  {
    LogError("The last dimension of multiplied tensors must be the same.");
    std::cout << "Dim LHS: ";
    PrintDims(Ldims);
    std::cout << "Dim RHS: ";
    PrintDims(Rdims);
    return {}; 
  }
  for (int i = 0; i < Ldims.size()-1; i++)
    new_dims.push_back(Ldims[i]);
  new_dims.push_back(Rdims[0]);


  return new_dims;
}






extern "C" float StoreDimsOnDemand(char *tensor_name, float d)
{
  std::vector<float> dims;
  
  if (NamedDims.count(tensor_name)>0)
    dims = NamedDims[tensor_name];

  dims.push_back(d);

  NamedDims[tensor_name] = dims;
  return 0;
}



extern "C" float CalculateIdxOffset(char *tensor_name, int first_idx, ...) {
  
  std::cout << "CalculateIdxOffset of " << tensor_name << "\n";

  DT_tensor *tensor = NamedTensorsT[tensor_name];


  // PrintDims(tensor->dims);

  std::vector<int> idxs, new_dims_no_minus, dims;
  int current_dims_prod;
  bool has_minus = false;
  dims = tensor->dims;

  int idx_at = 0;

  
  va_list args;
  va_start(args, first_idx);

  if (first_idx!=-1)
    new_dims_no_minus.push_back(first_idx);
  else
    has_minus=true;
  
    
  idxs.push_back(first_idx);

  dims = RemoveFirstDim(dims);
  
  current_dims_prod = DimsProd(dims);



  // std::cout << "---idx: " << first_idx << "|cur_dims_prod: " << std::to_string(current_dims_prod) << "|adding: " << std::to_string(current_dims_prod*first_idx) << ".\n";

  idx_at += (int)(current_dims_prod*first_idx);




  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions??? (calc idx)");
      return 0;
    }

    float idx = va_arg(args, float);
    if (idx==TERMINATE_VARARG)
      break;

    idxs.push_back(idx);
    
    dims = RemoveFirstDim(dims);
    
    current_dims_prod = DimsProd(dims);

    // std::cout << "---idx: " << idx << "|cur_dims_prod: " << std::to_string(current_dims_prod) << "|adding: " << std::to_string(current_dims_prod*idx) << ".\n";

    idx_at += (int)(current_dims_prod*idx);

    

    if (idx!=-1)
      new_dims_no_minus.push_back(idx);
    else
      has_minus=true;
  }
  va_end(args);



  return idx_at;
}


void broadcast_lastdim_add_backward2(float *dx, float *dy, int x_size, int y_size)
{

  int leading_dim = y_size/x_size;


  int grid_size, block_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(y_size);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];


  sum_over_last_dim_kernel<<<grid_size, block_size, 0, main_stream>>>(dy, dx, y_size, leading_dim);
}


extern "C" float tensor_shape(Scope_Struct *scope_struct, DT_tensor *tensor)
{
  std::cout << "\nTensor \033[95m" << tensor->name<<"/"<<tensor->scopeless_name << "\033[0m:\n   ";
  PrintDims(tensor->dims);

  return 0;
}