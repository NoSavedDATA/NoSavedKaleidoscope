#pragma once

#include "../../tensor/tensor_struct.h"




void MSEBackward(float *y_hat, float *y,
                 int dims_prod, 
                 float *dloss,
                 float scale);


extern "C" float mse(Tensor *y_hat, Tensor *y, float scale);



extern "C" void *mse_with_priorities(int thread_id, Tensor *y_hat, Tensor *y, float scale, Tensor *is_w);




void MSEWithPrioritiesBackward(Tensor *loss_tensor,
                 float *dloss);
