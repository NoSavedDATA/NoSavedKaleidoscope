
#pragma once

#include <limits>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>



constexpr int32_t TERMINATE_VARARG = -2147483647;
constexpr int32_t COPY_TO_END_INST = 0x7FADBEEF;

extern bool Shall_Exit;


extern std::map<std::string, std::string> functions_return_type, reverse_ops;

extern std::unordered_map<std::string, std::vector<int>> ClassPointers;
extern std::unordered_map<std::string, std::vector<std::string>> ClassPointersType;