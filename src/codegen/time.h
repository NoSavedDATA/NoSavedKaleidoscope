#pragma once

#include <chrono>

#include "../mangler/scope_struct.h"

unsigned int get_millisecond_time();

extern std::chrono::high_resolution_clock::time_point START_TIME;
