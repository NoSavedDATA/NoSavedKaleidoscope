#pragma once

#include <chrono>
#include"include.h"


extern std::chrono::high_resolution_clock::time_point START_TIME;

extern "C" float start_timer(float id);


extern "C" float end_timer(float id);


extern "C" void __slee_p_(float id);

extern "C" float silent_sleep(float id);

