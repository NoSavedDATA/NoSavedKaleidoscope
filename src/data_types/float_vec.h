#pragma once

#include <iostream>
#include <vector>


extern "C" float PrintFloatVec(std::vector<float> vec);

extern "C" void * zeros_vec(float size);

extern "C" void * ones_vec(float size);
