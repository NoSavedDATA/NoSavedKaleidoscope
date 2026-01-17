#include<map>
#include<vector>
#include<string>
#include<iostream>
#include<algorithm>
#include<random>
#include<memory>
#include<cstring>
#include<sstream>

bool ShallCodegen = true;

// Cuda
#include "include.h"




int round_nearest_pow2(int x) {
    if (x == 0) return 1; // By convention, or you can return 0

    int abs_x = std::abs(x);
    int exp = std::round(std::log2(abs_x));
    int result = 1 << exp;

    // Clamp to avoid overflow
    if (result < 0) result = std::numeric_limits<int>::max();

    return (x > 0) ? result : -result;
}





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

std::string remove_suffix(const std::string& input_string, std::string suffix) {
    if (input_string.length() >= suffix.length() && 
        input_string.substr(input_string.length() - suffix.length()) == suffix) {
        return input_string.substr(0, input_string.length() - suffix.length());
    }
    return input_string;
}

std::string erase_before_pattern(std::string s, std::string pattern) {
    size_t pos = s.find(pattern);
    if (pos != std::string::npos) {
        s.erase(0, pos + pattern.length());
    }
    return s;
}

std::string remove_substring(const std::string& str, const std::string& substr) {
    std::string result = str;  // Copy the original string
    size_t pos = result.find(substr);
    if (pos != std::string::npos) {
        result.erase(pos, substr.length());
    }
    return result;
}
bool starts_with(const char* str, const char* sub) {
  return strncmp(str, sub, strlen(sub)) == 0;
}

char *str_to_char(std::string str)
{
    char *c_str = new char[str.length() + 1]; // +1 for the null terminator
    std::strcpy(c_str, str.c_str());

    return c_str;
}





int count_pattern(const std::string& text, const std::string& pattern) {
  int count = 0;
  size_t pos = 0;

  std::cout << "Trying to count"  << "\n";
  // Iterate while finding occurrences of the pattern
  while ((pos = text.find(pattern, pos)) != std::string::npos) {
    count++;
    pos += pattern.length(); // Move to the character after the found pattern
  }

  return count;
}

std::vector<std::string> split_str(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream stream(str);

  while (std::getline(stream, token, delimiter)) {
    tokens.push_back(token);
  }

  return tokens;
}

std::vector<std::string> split(const char* input, const std::string& delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0, end = 0;
    while ((end = std::string(input + start).find(delimiter)) != std::string::npos) {
        tokens.push_back(std::string(input + start, end));
        start += end + delimiter.length();
    }
    tokens.push_back(std::string(input + start));
    return tokens;
}


bool in_char(char ch, const std::vector<char>& list) {
  // Use std::find to efficiently search the list for the character
  return std::find(list.begin(), list.end(), ch) != list.end();
}

bool in_str(std::string str, std::vector<std::string> list) {
    return std::find(list.begin(), list.end(), str) != list.end();
}

bool in_int(int value, const std::vector<int> list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}

bool in_int_ptr(int value, int *list, int size) {
  for (int i=0; i<size; ++i)
  {
    if (list[i]==value)
      return true;
  }
  return false;
}

bool in_float_vec(float value, const std::vector<float>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}

bool in_char_ptr_vec(const char *value, const std::vector<char *>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}
bool in_floatptr_vec(const float *value, const std::vector<float *>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}
bool in_int8ptr_vec(const int8_t* value, const std::vector<int8_t*>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}

std::vector<std::string> concat_str_vec(std::vector<std::string> l, std::vector<std::string>r)
{
  std::vector<std::string> concatenated_vectors = l;
  concatenated_vectors.insert(concatenated_vectors.end(), r.begin(), r.end());
  return concatenated_vectors;
}
std::vector<int> concat_int_vec(std::vector<int> l, std::vector<int>r)
{
  std::vector<int> concatenated_vectors = l;
  concatenated_vectors.insert(concatenated_vectors.end(), r.begin(), r.end());
  return concatenated_vectors;
}
