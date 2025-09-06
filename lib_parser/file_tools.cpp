#pragma once



#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>


#include "file_tools.h"

namespace fs = std::filesystem;



bool fexists(const std::string& filename) {
    fs::path filePath = filename;
    return std::filesystem::exists(filePath);
}

void fremove(const std::string &filename) {
    if (fs::exists(filename))
        fs::remove(filename);
}

bool was_file_modified(std::string lib_file, std::string parsed_file) {
    

    // Get cpp file time until last modification.
    fs::path file_path(lib_file);
    auto ftime = fs::last_write_time(file_path);


    // Get parsed lib time
    std::string first_line;
    std::ifstream parsed_lib_file(parsed_file);
    std::getline(parsed_lib_file, first_line);


    // Compare times
    auto sctp = std::chrono::system_clock::time_point(ftime.time_since_epoch());
    std::time_t cftime = std::chrono::system_clock::to_time_t(sctp);

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&cftime), "%F %T");
    std::string string_last_modified = oss.str();

    // std::cout << "Last modified: " << string_last_modified << ".\n";
    // std::cout << "First line " << first_line << ".\n";

    bool was_modified = string_last_modified!=first_line;
    
    // std::cout << "Was modified? " << std::to_string(was_modified) << ".\n";

    return was_modified;
}


std::vector<fs::path> glob_cpp(const fs::path& rootDir, std::string extension) {
    std::vector<fs::path> cppFiles;
    std::mutex mtx;

    for (const auto& entry : fs::recursive_directory_iterator(rootDir)) {
        const std::string filename = entry.path().string();
        if (filename.size() >= extension.size() &&
            filename.compare(filename.size() - extension.size(), extension.size(), extension) == 0) {
            std::lock_guard<std::mutex> lock(mtx);
            cppFiles.push_back(entry.path());
        }
    }

    return std::move(cppFiles);
}


std::string replace_slash_with_double_underscore(const std::string& input) {
    std::string output;
    for (char c : input) {
        if (c == '/') {
            output += "__";
        } else {
            output += c;
        }
    }
    return output;
}

void Lib_Files::Mangle_Lib_File_Name(std::string fname) {

    std::string path = fname;
    // Replace '/' with '_'
    // std::replace(path.begin(), path.end(), '/', '__');
    path = replace_slash_with_double_underscore(path);

    // Replace ".cpp" with ".txt"
    size_t pos = path.rfind(".cpp");
    if (pos != std::string::npos) {
        path.replace(pos, 4, "");
    }
    pos = path.rfind(".cu");
    if (pos != std::string::npos) {
        path.replace(pos, 3, "");
    }
    
    std::string root = "lib_parser/parsed_libs/";
    
    file_name = root + path + "_llvm_lib.txt";
    returns_dict = root + path + "_returns_dict.txt";
    returns_data_dict = root + path + "_returns_data_dict.txt";
    args_dict = root + path + "_args_dict.txt";
    user_cpp = root + path + "_user_cpp.txt";
    clean_up = root + path + "_clean_up.txt";
    backward = root + path + "_backward.txt";

    // std::cout << file_name << "   /   " << user_cpp << "   /   " << returns_dict << ".\n";
}

std::string Mangle_Lib_File_Name(std::string fname) {

    std::string path = fname;
    // Replace '/' with '_'
    // std::replace(path.begin(), path.end(), '/', '__');
    path = replace_slash_with_double_underscore(path);

    // Replace ".cpp" with ".txt"
    size_t pos = path.rfind(".cpp");
    if (pos != std::string::npos) {
        path.replace(pos, 4, "_llvm_lib.txt");
    }
    pos = path.rfind(".cu");
    if (pos != std::string::npos) {
        path.replace(pos, 3, "_llvm_lib.txt");
    }
    
    path = "lib_parser/parsed_libs/" + path;
    return path;
}




std::string Demangle_File_Name(std::string input) {


    size_t last_slash = input.find_last_of('/');
    std::string filename = (last_slash != std::string::npos) ? input.substr(last_slash + 1) : input;

    // Step 2: Remove the suffix "llvm_lib.txt"
    const std::string suffix = "_llvm_lib.txt";
    if (filename.size() >= suffix.size() &&
        filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0) {
        filename.erase(filename.size() - suffix.size());
    }

    // Step 3: Replace all "__" with "/"
    std::string result;
    for (size_t i = 0; i < filename.size(); ++i) {
        if (i + 1 < filename.size() && filename[i] == '_' && filename[i + 1] == '_') {
            result += '/';
            ++i; // Skip next '_'
        } else {
            result += filename[i];
        }
    }

    return result;
}