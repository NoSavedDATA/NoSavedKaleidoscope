#pragma once

#include <vector>

#include "calculate_grids.h"





std::vector<int> CalculateGridAndBlockSizes(int dims_prod, int pre_block_size=-1);



std::vector<int> CalculateSimpleWarpGridAndBlockSizes(int B);
