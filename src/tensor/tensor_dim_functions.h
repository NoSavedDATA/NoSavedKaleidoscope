
#pragma once


extern "C" void PrintDims(std::vector<float> dims);

int DimsProd(std::vector<float> dims);

std::vector<float> BatchLessDims(std::vector<float> dims);

std::vector<float> RemoveLastDim(std::vector<float> dims);

std::vector<float> RemoveFirstDim(std::vector<float> dims);