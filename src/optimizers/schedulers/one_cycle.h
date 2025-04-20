#pragma once


#include "../../mangler/scope_struct.h"

extern "C" float OneCycleLR(Scope_Struct *scope_struct, float base_lr, float step, float max_steps);
