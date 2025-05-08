#pragma once

#include "../../tensor/tensor_struct.h"




void MSEBackward(float *y_hat, float *y,
                 int dims_prod, 
                 float *dloss,
                 float scale);


extern "C" float mse(DT_tensor *y_hat, DT_tensor *y, float scale);



extern "C" void *mse_with_priorities(int thread_id, DT_tensor *y_hat, DT_tensor *y, float scale, DT_tensor *is_w);




void MSEWithPrioritiesBackward(DT_tensor *loss_tensor,
                 float *dloss);
