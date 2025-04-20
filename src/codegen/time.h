#pragma once

#include <chrono>

#include "../mangler/scope_struct.h"
#include"include.h"


extern std::chrono::high_resolution_clock::time_point START_TIME;

extern "C" float start_timer(Scope_Struct *scope_struct, float id);


extern "C" float end_timer(Scope_Struct *scope_struct, float id);


extern "C" void __slee_p_(Scope_Struct *scope_struct, float id);

extern "C" float silent_sleep(Scope_Struct *scope_struct, float id);

