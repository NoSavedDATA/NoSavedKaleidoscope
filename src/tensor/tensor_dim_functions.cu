
// #include <algorithm>
// #include <cstdarg>
// #include <cassert>
// #include <cctype>
// #include <cstring>
// #include <cstdint>
// #include <cstdio>
// #include <cstdlib>
// #include <map>
// #include <memory>
// #include <string>
// #include <iostream>
// #include <numeric>
// #include <utility>
// #include <vector>
// #include <iomanip>
// #include <math.h>
// #include <fenv.h>
// #include <tuple>
// #include <glob.h>
// #include <chrono>
// #include <thread>
// #include <random>
// #include <float.h>
// #include <fstream>
// #include <sstream>
// #include <filesystem>
// #include <stdio.h>
// #include <stdlib.h>
// #include <omp.h>

// #include <cuda_fp16.h>

// #include "../common/include.h"
// #include "../cu_commons.h"
// #include "include.h"



// extern "C" void PrintDims(std::vector<float> dims)
// {
//   std::cout << "dims: [";
//   for (int i=0; i<dims.size();i++)
//   {
//     std::cout << (int)dims[i];
//     if (i==dims.size()-1)
//       std::cout << "]";
//     else
//       std::cout << ", ";
//   }
//   std::cout  << "\n";
// }
// int DimsProd(std::vector<float> dims)
// {
//   if (dims.size()==1)
//     return (int) dims[0];

//   float aux=1;
//   for (int i = 0; i < dims.size(); i++)
//     aux = aux*dims[i];
//   return (int)aux;
// }

// std::vector<float> BatchLessDims(std::vector<float> dims)
// {
//   // Removes first dim (batch dim).
//   if (dims.size()<=1)
//     LogError("Cannot remove the batch dimension of a unidimensional tensor.");

//   std::vector<float> new_dims;

//   for (int i=0; i<dims.size()-1;i++)
//     new_dims.push_back(dims[i+1]);

//   return new_dims;
// }

// std::vector<float> RemoveLastDim(std::vector<float> dims)
// {
//   // Removes first dim (batch dim).
//   if (dims.size()<=1)
//   {
//     return {1.0f};
//     //LogError("Cannot remove the batch dimension of a unidimensional tensor.");
//   }

//   std::vector<float> new_dims;

//   for (int i=0; i<dims.size()-1;i++)
//     new_dims.push_back(dims[i]);

//   return new_dims;
// }

// std::vector<float> RemoveFirstDim(std::vector<float> dims)
// {
//   // Removes first dim (batch dim).
//   if (dims.size()<=1)
//     return {1.0f};

//   std::vector<float> new_dims;

//   for (int i=0; i<dims.size()-1;i++)
//     new_dims.push_back(dims[i+1]);

//   return new_dims;
// }