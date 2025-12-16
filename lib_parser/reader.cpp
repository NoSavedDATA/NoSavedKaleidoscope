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
    files = aux;

    

    has_started = true;
}





std::string current_file_name = "";


char get_file_char() {
    if(!has_started)
    {
        // std::cout << "Reading first file" << ".\n";
        std::string folder = "src";
        get_cpp_files(folder);
        if(files.size()==0)
            return tok_finish;
        current_file_name = files[0].string();
        // std::cout << "Reading file: " << current_file_name << ".\n";
        file.open(current_file_name);
        file_counter+=1;
    }

    char ch;
    if(file.get(ch))
    {
        FileRead = FileRead + ch;
        return ch;
    }
    else {
        FileRead="";
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

