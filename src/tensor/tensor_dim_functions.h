#pragma once

#include <vector>

#include "../mangler/scope_struct.h"
#include "tensor_struct.h"




extern "C" void PrintDims(std::vector<int> dims);

int DimsProd(std::vector<float> dims);
int DimsProd(std::vector<int> dims);

std::vector<float> BatchLessDims(std::vector<float> dims);
std::vector<int> BatchLessDims(std::vector<int> dims);

std::vector<float> RemoveLastDim(std::vector<float> dims);
std::vector<float> RemoveFirstDim(std::vector<float> dims);

std::vector<int> RemoveLastDim(std::vector<int> dims);
std::vector<int> RemoveFirstDim(std::vector<int> dims);



std::vector<float> format_BatchFirst_Dims(std::vector<float> dims);
std::vector<int> format_BatchFirst_Dims(std::vector<int> dims);


std::vector<float> format_LinearLayer_Dims(std::vector<float> dims);
std::vector<int> format_LinearLayer_Dims(std::vector<int> dims);


int resultingDimsProdOnMult(std::vector<float> Ldims, std::vector<float> Rdims);
int resultingDimsProdOnMult(std::vector<int> Ldims, std::vector<int> Rdims);

std::vector<float> NewDimsOnMult(std::vector<float> Ldims, std::vector<float> Rdims);
std::vector<int> NewDimsOnMult(std::vector<int> Ldims, std::vector<int> Rdims);



void broadcast_lastdim_add_backward2(float *dx, float *dy, int x_size, int y_size);




