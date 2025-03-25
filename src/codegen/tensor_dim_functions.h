#pragma once

#include <vector>


extern "C" void PrintDims(std::vector<float> dims);

int DimsProd(std::vector<float> dims);

std::vector<float> BatchLessDims(std::vector<float> dims);

std::vector<float> RemoveLastDim(std::vector<float> dims);

std::vector<float> RemoveFirstDim(std::vector<float> dims);



std::vector<float> format_BatchFirst_Dims(std::vector<float> dims);


std::vector<float> format_LinearLayer_Dims(std::vector<float> dims);


int resultingDimsProdOnMult(std::vector<float> Ldims, std::vector<float> Rdims);

std::vector<float> NewDimsOnMult(std::vector<float> Ldims, std::vector<float> Rdims);


extern "C" void *NewDimsOnIdx(std::vector<float> dims);

extern "C" float StoreDimsOnDemand(char *tensor_name, float d);
