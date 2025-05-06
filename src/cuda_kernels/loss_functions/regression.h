#pragma once

#include "../../tensor/tensor_struct.h"




void MSEBackward(float *y_hat, float *y,
                 int dims_prod, 
                 float *dloss,
                 float scale);


extern "C" float mse(data_type_tensor *y_hat, data_type_tensor *y, float scale);



extern "C" void *mse_with_priorities(int thread_id, data_type_tensor *y_hat, data_type_tensor *y, float scale, data_type_tensor *is_w);




void MSEWithPrioritiesBackward(data_type_tensor *loss_tensor,
                 float *dloss);
