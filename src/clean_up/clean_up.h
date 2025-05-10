#pragma once

#include <functional>
#include <map>
#include <string>

extern std::map<std::string, std::function<void(void *)>> clean_up_functions;