#pragma once

#include "../../mangler/scope_struct.h"

extern "C" float CosineLR(Scope_Struct *scope_struct, float base_lr, float min_lr, float step, float max_steps);
