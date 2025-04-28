#pragma once

#include <vector>

#include "calculate_grids.h"





std::vector<int> CalculateGridAndBlockSizes(int dims_prod, int pre_block_size=-1);
void CalculateGridAndBlockSizes(int dims_prod, int& grid_size, int& block_size, int pre_block_size = -1);
void CalculateGridAndBlockSizes(int dims_prod, int& grid_size, int& block_size, int& shared_mem_size, int pre_block_size = -1);
void CalculateSimpleWarpGridAndBlockSizes(int B, int &grid_size, int &block_size);


std::vector<int> CalculateSimpleWarpGridAndBlockSizes(int B);
