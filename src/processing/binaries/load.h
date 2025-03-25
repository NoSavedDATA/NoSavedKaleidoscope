#pragma once

#include "../../tensor/include.h"

extern "C" float load_bin(Tensor *tensor, char *bin_name);

extern "C" float load_bin_idx(Tensor *tensor, char *bin_name, float first_idx, ...);

extern "C" float wload_bin(Tensor *tensor, char *bin_name, float worker_idx, float batch_idx);