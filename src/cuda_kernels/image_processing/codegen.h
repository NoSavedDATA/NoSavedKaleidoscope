#pragma once

#include "../../tensor/tensor_struct.h"


extern "C" void *RandomCrop(int thread_id, Tensor *tensor, float padding);






extern "C" void *RandomHorizontalFlip(int thread_id, Tensor *tensor);






extern "C" void *NormalizeImg(int thread_id, Tensor *tensor, Tensor *mean, Tensor *std);



extern "C" void *Jitter(int thread_id, Tensor *tensor, float factor);
