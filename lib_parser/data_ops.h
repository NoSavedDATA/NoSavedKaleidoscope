#pragma once

#include <string>


bool ends_with(std::string str_input, std::string str_end);

bool begins_with(const std::string& str_input, const std::string& str_start); 
bool contains_str(const std::string& str_input, const std::string& str_sub); 
std::string remove_substring(const std::string& str, const std::string& substr); 