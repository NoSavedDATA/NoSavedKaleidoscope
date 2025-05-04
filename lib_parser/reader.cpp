#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "tokenizer.h"



namespace fs = std::filesystem;


std::ifstream file;
bool has_started=false;

std::vector<fs::path> files;

int file_counter = 0;



std::vector<fs::path> glob_cpp(const fs::path& rootDir) {
    std::vector<fs::path> cppFiles;
    std::mutex mtx;

    for (const auto& entry : fs::recursive_directory_iterator(rootDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".cpp") {
            std::lock_guard<std::mutex> lock(mtx); // Thread-safe if parallelized
            cppFiles.push_back(entry.path());
        }
    }

    return std::move(cppFiles);
}




void get_cpp_files(const fs::path& rootDir) {


    files = glob_cpp(rootDir);

    // std::vector<fs::path> aux;
    // aux.push_back(files[0]);
    // files = aux;


    std::vector<fs::path> aux;

    for (auto &lib_file : files){
        std::string fname = lib_file.string()
        std::string parsed_lib = Mangle_Lib_File_Name(fname);
        
        file.open(fname);

        file.close();
    }


    
    std::exit(0);

    // std::cout << "files[0]" << files[0] << ".\n";

    has_started = true;
}





std::string current_file_name = "";


char get_file_char() {

    if(!has_started)
    {
        // std::cout << "Reading first file" << ".\n";
        std::string folder = "src/data_types";
        get_cpp_files(folder);
        // std::cout << "Got files. Reading" << ".\n";
        current_file_name = files[0].string();
        std::cout << "Reading file: " << current_file_name << ".\n";
        file.open(current_file_name);
        file_counter+=1;
    }




    char ch;
    if(file.get(ch))
        return ch;
    else {
        file.close();
        
        std::cout << "\n\n\n\n=====================================================================================\n\n\n\n\n";
        std::cout << "Finished reading file" << current_file_name << ".\n";
        
        if(file_counter>=files.size())
            return tok_finish;
       
        current_file_name = files[file_counter].string();

        std::cout << "open " << current_file_name << ".\n";
        file.open(current_file_name);

        file_counter+=1;

        return tok_eof;
    }
}

