#pragma once

#include <string>
#include <map>
#include <vector>
#include <iostream>


#include "../common/extension_functions.h"
#include "../threads/include.h"


extern std::map<size_t, std::vector<char *>> CharPool;


void move_to_char_pool(size_t length, char *char_ptr, std::string from);

char *get_from_char_pool(size_t length, std::string from);
