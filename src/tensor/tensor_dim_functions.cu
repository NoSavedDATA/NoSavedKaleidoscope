

#include<vector>
#include<map>
#include<iostream>
#include<random>
#include<thread>
#include<string>

#include "../common/include.h"
#include "../compiler_frontend/logging.h"
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