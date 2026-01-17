#pragma once

#include <vector>
#include <map>
#include <string>






bool ends_with(std::string str_input, std::string str_end);

bool begins_with(const std::string& str_input, const std::string& str_start);

bool contains_str(const std::string& str_input, const std::string& str_sub);


std::string remove_suffix(const std::string& input_string, std::string suffix);

std::string remove_substring(const std::string& str, const std::string& substr);

bool starts_with(const char* str, const char* sub);


std::string erase_before_pattern(std::string s, std::string pattern);

char *str_to_char(std::string str);





int count_pattern(const std::string& text, const std::string& pattern);

std::vector<std::string> split_str(const std::string& str, char delimiter);

std::vector<std::string> split(const char* input, const std::string& delimiter);


bool in_char(char ch, const std::vector<char>& list);

bool in_str(std::string str, std::vector<std::string> list);

bool in_int(int value, const std::vector<int> list);

bool in_int_ptr(int value, int *list, int size);

bool in_float_vec(float value, const std::vector<float>& list);

bool in_char_ptr_vec(const char *value, const std::vector<char *>& list);

bool in_floatptr_vec(const float *value, const std::vector<float *>& list);

bool in_int8ptr_vec(const int8_t* value, const std::vector<int8_t*>& list);

std::vector<std::string> concat_str_vec(std::vector<std::string> l, std::vector<std::string> r);

std::vector<int> concat_int_vec(std::vector<int> l, std::vector<int>r);

extern bool ShallCodegen;



int round_nearest_pow2(int x); 
