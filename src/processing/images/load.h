#pragma once

extern float *current_data;

#include "../../mangler/scope_struct.h"
#include "../../tensor/include.h"

extern "C" float *load_img(Scope_Struct *scope_struct,char *img_name);

extern "C" float * gload_img(Scope_Struct *scope_struct,Tensor tensor, char *img_name, float batch_idx);

extern "C" float * wload_img(Scope_Struct *scope_struct,Tensor *tensor, char *img_name, float worker_idx, float batch_idx);


extern "C" float * wload_img_resize(Scope_Struct *scope_struct,Tensor *tensor, char *img_name, float worker_idx, float batch_idx, float c, float h, float w);


extern "C" float load_preprocess_img(Scope_Struct *scope_struct,Tensor tensor, char *img_name);
