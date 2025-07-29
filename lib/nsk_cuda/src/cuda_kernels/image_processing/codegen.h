#pragma once

#include "../../tensor/tensor_struct.h"


extern "C" void *RandomCrop(int thread_id, DT_tensor *tensor, float padding);






extern "C" void *RandomHorizontalFlip(int thread_id, DT_tensor *tensor);






extern "C" void *NormalizeImg(int thread_id, DT_tensor *tensor, DT_tensor *mean, DT_tensor *std);



extern "C" void *Jitter(int thread_id, DT_tensor *tensor, float factor);
