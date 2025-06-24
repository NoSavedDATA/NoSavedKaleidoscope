#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers> 
#include <memory> 
#include <cstring> 


#include "../common/cu_commons.h"
#include "../mangler/scope_struct.h"
#include "include.h"


extern "C" float PrintTensor(Scope_Struct *scope_struct, DT_tensor *tensor){
  int thread_id = scope_struct->thread_id;
  std::string tensorName = tensor->name;
  std::cout << "Printing tensor " << tensorName << " at stream " << thread_id << "\n";



  int arr_size = tensor->dims_prod;
  float *tensor_cpu = new float[arr_size];

  
  std::vector<int> dims = tensor->dims;
  
  
  cudaStream_t stream = ThreadsStream[thread_id];
  tensor->Sync();
  cudaStreamSynchronize(stream);
  cudaDeviceSynchronize();
  cudaCheck(cudaMemcpy(tensor_cpu, tensor->tensor_ptr, arr_size*sizeof(float), cudaMemcpyDeviceToHost));


  std::cout << "\nTensor \033[95m" << tensorName << "\033[0m:\n\n";
  PrintDims(dims);
  std::cout << "\n";
  std::vector<int> ends;


  for (int i = 0; i < dims.size(); i++) {
    int prod=1;
    for (int j = 0; j <= i; j++)
      prod = prod*dims[dims.size()-1-j];
    ends.push_back(prod);
  }


  int line = 1;
  bool line_changed = true;

  
  //if (arr_size>2000)
  //  arr_size = 2000;

  for (int i = 0; i < arr_size; i++) {

    int to_prints = 0;

    for (int e = 0; e < ends.size(); e++)
    {
      if (fmod((arr_size-i),(int)ends[e]) == 0.0f)
        to_prints+=1;
    }

    if(to_prints>0)
    {
      for (int j=0; j<(dims.size()-to_prints); j++)
        std::cout << " ";
        
      for (int j=0; j<to_prints; j++)
        std::cout << "[";
    }
    

    //std::cout << "LAST SIZE " << dims[dims.size()-1] << " Mod: " << fmod(i, 1+dims[dims.size()-1]) << "\n";
    int precision;
    if (tensor_cpu[i]>=0)
      precision=4;
    else
      precision=3;
    std::cout << std::fixed  << std::setprecision(precision) << tensor_cpu[i];


    for (int e = 0; e < ends.size(); e++)
      if (fmod((i+1), ends[e]) == 0.0f)
        std::cout << "],";
    

    if (i!=(arr_size-1))
    {
      if (fmod(i+1, dims[dims.size()-1]) == 0.0f)
      {
        line+=1;
        line_changed=true;
        std::cout << "\n";
      }
      else
        std::cout << ",  ";
    }

    if(fmod(i+1, ends[1]) == 0.0f)
      std::cout << "\n";


  }
  
  std::cout << "\n";
  PrintDims(dims);
  std::cout << "\n\n";

  delete[] tensor_cpu;

  return 0;
}





extern "C" float PrintTensorF(const float *cuda_tensor, int d1, int d2){

  std::vector<int> dims;
  dims.push_back(d1);
  dims.push_back(d2);

  int arr_size = DimsProd(dims);


  float *tensor = new float[arr_size];
  //std::cout << "Printing DT_tensor " << arr_size << "\n";
  
  cudaDeviceSynchronize();
  cudaCheck(cudaMemcpy(tensor, cuda_tensor, arr_size*sizeof(float), cudaMemcpyDeviceToHost));


  
  std::cout << "\n";
  PrintDims(dims);
  std::vector<float> ends;


  for (int i = 0; i < dims.size(); i++) {
    int prod=1;
    for (int j = 0; j <= i; j++)
      prod = prod*dims[dims.size()-1-j];
    ends.push_back(prod);
  }


  int line = 1;
  bool line_changed = true;
  for (int i = 0; i < arr_size; i++) {

    int to_prints = 0;

    for (int e = 0; e < ends.size(); e++)
    {
      if (fmod((arr_size-i),(int)ends[e]) == 0.0f)
        to_prints+=1;
    }

    if(to_prints>0)
    {
      for (int j=0; j<(dims.size()-to_prints); j++)
        std::cout << " ";
        
      for (int j=0; j<to_prints; j++)
        std::cout << "[";
    }
    

    //std::cout << "LAST SIZE " << dims[dims.size()-1] << " Mod: " << fmod(i, 1+dims[dims.size()-1]) << "\n";
    int precision;
    if (tensor[i]>=0)
      precision=4;
    else
      precision=3;
    std::cout << std::fixed  << std::setprecision(precision) << tensor[i];


    for (int e = 0; e < ends.size(); e++)
      if (fmod((i+1),(int)ends[e]) == 0.0f)
        std::cout << "],";
    

    if (i!=(arr_size-1))
    {
      if (fmod(i+1, dims[dims.size()-1]) == 0.0f)
      {
        line+=1;
        line_changed=true;
        std::cout << "\n";
      }
      else
        std::cout << ",  ";
    }

    if(fmod(i+1, ends[1]) == 0.0f)
      std::cout << "\n";


  }
  std::cout << "\n";
  
  delete[] tensor;

  return 0;
}




extern "C" float PrintTensorI8(const int8_t *cuda_tensor, int d1, int d2){

  std::vector<int> dims;
  dims.push_back(d1);
  dims.push_back(d2);

  int arr_size = DimsProd(dims);


  int8_t *tensor = new int8_t[arr_size];
  //std::cout << "Printing DT_tensor " << arr_size << "\n";
  
  cudaDeviceSynchronize();
  cudaCheck(cudaMemcpy(tensor, cuda_tensor, arr_size*sizeof(int8_t), cudaMemcpyDeviceToHost));


  
  std::cout << "\n";
  PrintDims(dims);
  std::vector<int> ends;


  for (int i = 0; i < dims.size(); i++) {
    int prod=1;
    for (int j = 0; j <= i; j++)
      prod = prod*dims[dims.size()-1-j];
    ends.push_back(prod);
  }


  int line = 1;
  bool line_changed = true;
  for (int i = 0; i < arr_size; i++) {

    int to_prints = 0;

    for (int e = 0; e < ends.size(); e++)
    {
      if (fmod((arr_size-i),(int)ends[e]) == 0)
        to_prints+=1;
    }

    if(to_prints>0)
    {
      for (int j=0; j<(dims.size()-to_prints); j++)
        std::cout << " ";
        
      for (int j=0; j<to_prints; j++)
        std::cout << "[";
    }
    

    std::cout << (int)tensor[i];


    for (int e = 0; e < ends.size(); e++)
      if (fmod((i+1),(int)ends[e]) == 0)
        std::cout << "],";
    

    if (i!=(arr_size-1))
    {
      if (fmod(i+1, dims[dims.size()-1]) == 0)
      {
        line+=1;
        line_changed=true;
        std::cout << "\n";
      }
      else
        std::cout << ",  ";
    }

    if(fmod(i+1, ends[1]) == 0)
      std::cout << "\n";


  }
  std::cout << "\n";
  
  delete[] tensor;

  return 0;
}