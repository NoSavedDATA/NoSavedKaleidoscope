#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "include.h"


namespace fs = std::filesystem;


std::ifstream file;
bool has_started=false;

std::vector<fs::path> files;

int file_counter = 0;


bool fexists(const std::string& filename) {
    std::filesystem::path filePath = filename;
    return std::filesystem::exists(filePath);
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
        if (entry.is_regular_file() && entry.path().extension() == extension) {
            std::lock_guard<std::mutex> lock(mtx); // Thread-safe if parallelized
            cppFiles.push_back(entry.path());
        }
    }

    return std::move(cppFiles);
}




void get_cpp_files(const fs::path& rootDir) {


    files = glob_cpp(rootDir);
    std::vector<fs::path> cu_files = glob_cpp(rootDir, ".cu");
    for (auto &cu_file : cu_files)
        files.push_back(cu_file);

    // std::vector<fs::path> aux;
    // aux.push_back(files[0]);
    // files = aux;


    std::vector<fs::path> aux;

    for (auto &lib_file : files){
        std::string fname = lib_file.string();
        std::string parsed_lib = Mangle_Lib_File_Name(fname);

        if (fexists(parsed_lib))
        {
            // std::cout << "FOUND FILE " << parsed_lib << ".\n";            
            if (was_file_modified(fname, parsed_lib))
                aux.push_back(lib_file);
        } else
            aux.push_back(lib_file);
    }

    // aux.push_back(files[0]);
    // files = aux;

    

    has_started = true;
}





std::string current_file_name = "";


char get_file_char() {

    if(!has_started)
    {
        // std::cout << "Reading first file" << ".\n";
        std::string folder = "src/data_types";
        get_cpp_files(folder);
        if(files.size()==0)
            return tok_finish;
        // std::cout << "Got files. Reading" << ".\n";
        current_file_name = files[0].string();
        // std::cout << "Reading file: " << current_file_name << ".\n";
        file.open(current_file_name);
        file_counter+=1;
    }




    char ch;
    if(file.get(ch))
        return ch;
    else {
        file.close();
        
        // std::cout << "\n\n\n\n=====================================================================================\n\n\n\n\n";
        // std::cout << "Finished reading file" << current_file_name << ".\n";
        
        if(file_counter>=files.size())
            return tok_finish;
       
        current_file_name = files[file_counter].string();

        // std::cout << "open " << current_file_name << ".\n";
        file.open(current_file_name);

        file_counter+=1;

        return tok_eof;
    }
}

