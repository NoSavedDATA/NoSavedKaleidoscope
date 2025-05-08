#pragma once

#include "../../tensor/include.h"

extern "C" float load_bin(DT_tensor *tensor, char *bin_name);

extern "C" float load_bin_idx(DT_tensor *tensor, char *bin_name, float first_idx, ...);

extern "C" float wload_bin(DT_tensor *tensor, char *bin_name, float worker_idx, float batch_idx);