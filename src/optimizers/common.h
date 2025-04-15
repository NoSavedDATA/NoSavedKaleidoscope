#pragma once


#include "interface.h"

std::unique_ptr<Optimizer> optimize(std::unique_ptr<Optimizer> optimizer);

extern std::unique_ptr<Optimizer> optimizer;