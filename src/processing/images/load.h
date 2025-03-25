#pragma once



#include "../../tensor/include.h"


extern "C" float *load_img(char *img_name);

extern "C" float * gload_img(Tensor tensor, char *img_name, float batch_idx);

extern "C" float * wload_img(Tensor *tensor, char *img_name, float worker_idx, float batch_idx);


extern "C" float * wload_img_resize(Tensor *tensor, char *img_name, float worker_idx, float batch_idx, float c, float h, float w);

