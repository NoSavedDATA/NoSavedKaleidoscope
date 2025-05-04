#pragma once



#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>


namespace fs = std::filesystem;

extern std::vector<fs::path> files;


extern std::ifstream file;


char get_file_char();

std::vector<fs::path> glob_cpp(const fs::path& rootDir, std::string extension=".cpp");

extern std::string current_file_name;

