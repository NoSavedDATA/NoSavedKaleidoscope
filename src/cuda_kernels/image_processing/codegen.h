#pragma once

#include "../../tensor/tensor_struct.h"


extern "C" void *RandomCrop(int thread_id, data_type_tensor *tensor, float padding);






extern "C" void *RandomHorizontalFlip(int thread_id, data_type_tensor *tensor);






extern "C" void *NormalizeImg(int thread_id, data_type_tensor *tensor, data_type_tensor *mean, data_type_tensor *std);



extern "C" void *Jitter(int thread_id, data_type_tensor *tensor, float factor);
