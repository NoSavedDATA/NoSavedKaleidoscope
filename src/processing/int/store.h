#pragma once

#include "../../tensor/tensor_struct.h"

extern "C" float save_as_int(int thread_id, Tensor *tensor, char *bin_name);
