#pragma once

#include "../../mangler/scope_struct.h"
#include "../../tensor/include.h"


extern "C" float save_img(Scope_Struct *scope_struct,int thread_id, Tensor *tensor, char *img_name);

