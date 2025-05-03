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


void get_cpp_files(const fs::path& rootDir) {
    std::vector<fs::path> cppFiles;
    std::mutex mtx;

    for (const auto& entry : fs::recursive_directory_iterator(rootDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".cpp") {
            std::lock_guard<std::mutex> lock(mtx); // Thread-safe if parallelized
            files.push_back(entry.path());
        }
    }

    has_started = true;
}



// void Parse_File(const std::string &filename) {

//     std::ifstream file(filename);
//     if (!file) {
//         std::cerr << "Failed to open file: " << filename << '\n';
//         return;
//     }


//     char ch;
//     while (file.get(ch))
//         Tokenize(ch);

// }

// int () {

//     std::string folder = "src/data_types";
//     auto files = get_cpp_files(folder);

//     for (const auto& file : files) {
//         std::cout << "Parsing lib file: " << file << '\n';
//         Parse_File(file.string());

//         return 0;
//     }

//     return 0;
// }





char get_file_char() {

    if(!has_started)
    {
        // std::cout << "Reading first file" << ".\n";
        std::string folder = "src/data_types";
        get_cpp_files(folder);
        // std::cout << "Got files. Reading" << ".\n";
        std::cout << "Reading file: " << files[0].string() << ".\n";
        file.open(files[0].string());
        file_counter+=1;
    }

    // std::cout << "checking file eof" << ".\n";

    // if(file.eof())
    // {
    //     // std::exit(0);
    //     std::cout << "\n\n\n\n=====================================================================================\n\n\n\n\n";
    //     std::cout << "Finished reading file" << files[file_counter-1].string() << ".\n";
    //     file_counter+=1;
    //     if(file_counter>files.size())
    //         return -255;
    //     file.open(files[file_counter].string());
    // }

    // std::cout << "getting char" << ".\n";

    char ch;
    if(file.get(ch))
        return ch;
    else {
        file.close();
        
        std::cout << "\n\n\n\n=====================================================================================\n\n\n\n\n";
        std::cout << "Finished reading file" << files[file_counter-1].string() << ".\n";
        
        if(file_counter>=files.size())
        {
            std::cout << "RETURNING TOKEN FINISH HERERERERE" << ".\n";
            return tok_finish;
        }
        

        std::cout << "open " << files[file_counter].string() << ".\n";
        file.open(files[file_counter].string());

        file_counter+=1;

        std::cout << "RETURNING TOKEN EOF HERERERERE" << ".\n";
        return tok_eof;
    }
}

