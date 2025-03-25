#pragma once

#include "../../tensor/tensor_struct.h"

extern "C" float save_as_bin(int thread_id, Tensor *tensor, char *bin_name);
