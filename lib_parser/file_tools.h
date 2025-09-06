#pragma once



#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>


namespace fs = std::filesystem;



struct Lib_Files {
    std::string file_name="";
    std::string user_cpp="";
    std::string returns_dict="";
    std::string returns_data_dict="";
    std::string args_dict="";
    std::string clean_up="";
    std::string backward="";

    void Mangle_Lib_File_Name(std::string);
};



std::string Mangle_Lib_File_Name(std::string fname);


bool fexists(const std::string& filename);

void fremove(const std::string &filename);

bool was_file_modified(std::string lib_file, std::string parsed_file); 

std::vector<fs::path> glob_cpp(const fs::path& rootDir, std::string extension); 

std::string Demangle_File_Name(std::string fname);