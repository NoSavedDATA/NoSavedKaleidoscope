#pragma once

#include <algorithm>
#include <string>


bool ends_with(std::string str_input, std::string str_end)
{
  return str_input.size() >= str_end.size() && str_input.compare(str_input.size() - str_end.size(), str_end.size(), str_end) == 0;
}
bool begins_with(const std::string& str_input, const std::string& str_start) {
    return str_input.size() >= str_start.size() && str_input.compare(0, str_start.size(), str_start) == 0;
}
bool contains_str(const std::string& str_input, const std::string& str_sub) {
    return str_input.find(str_sub) != std::string::npos;
}
std::string remove_substring(const std::string& str, const std::string& substr) {
    std::string result = str;  // Copy the original string
    size_t pos = result.find(substr);
    if (pos != std::string::npos) {
        result.erase(pos, substr.length());
    }
    return result;
}