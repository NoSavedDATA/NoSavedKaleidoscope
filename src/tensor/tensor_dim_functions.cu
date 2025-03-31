
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
#include "../data_types/include.h"
#include "../tensor/include.h"
#include "include.h"



extern "C" void PrintDims(std::vector<float> dims)
{
  std::cout << "dims: [";
  for (int i=0; i<dims.size();i++)
  {
    std::cout << (int)dims[i];
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



int resultingDimsProdOnMult(std::vector<float> Ldims, std::vector<float> Rdims)
{
  float aux=1;
  for (int i = 0; i < Ldims.size()-1; i++)
    aux = aux * Ldims[i];
  aux = aux * Rdims[0];
  return (int)aux;
}


std::vector<float> NewDimsOnMult(std::vector<float> Ldims, std::vector<float> Rdims)
{
  

  std::vector<float> new_dims;
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



extern "C" void *NewDimsOnIdx(std::vector<float> dims)
{
  std::vector<float> new_dims;

  for (int i = 0; i < dims.size()-1; i++)
    new_dims.push_back(dims[i+1]);


  // Aux to not lose pointers
  std::string random_str = RandomString(15);
  NamedDims[random_str] = new_dims; // Deal with new_dims being deleted after scope finished.
  AuxRandomStrs[random_str] = "dim";


  std::cout << "NewDimsOnIdx" << "\n";
  PrintDims(NamedDims[random_str]);

  return &NamedDims[random_str];
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



extern "C" float CalculateIdxOffset(char *tensor_name, float first_idx, ...) {
  
  //std::cout << "CalculateIdxOffset of " << tensor_name << "\n";

  Tensor *tensor = NamedTensorsT[tensor_name];

  std::vector<float> idxs, new_dims_no_minus, dims;
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

  idx_at += (int)(current_dims_prod*first_idx);



  //std::cout << "Get idx of " << tensor_name << "\nCalculateIdxOffset pushing dim: " << first_idx << "\n";

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

    idx_at += (int)(current_dims_prod*idx);

    //std::cout << "CalculateIdxOffset pushing dim: " << idx << "\n";
    

    if (idx!=-1)
      new_dims_no_minus.push_back(idx);
    else
      has_minus=true;
  }
  va_end(args);



  return idx_at;
}