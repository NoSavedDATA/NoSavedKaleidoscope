#pragma once

#include "../../tensor/tensor_struct.h"

extern "C" float save_as_bin(int thread_id, data_type_tensor *tensor, char *bin_name);
