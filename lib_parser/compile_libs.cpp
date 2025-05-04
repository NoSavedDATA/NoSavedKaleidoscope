#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "include.h"

namespace fs = std::filesystem;

std::string Get_Base_Lib() {
    std::string fpath = "src/libs_llvm/libs.cpp";


    std::ifstream libs_file(fpath);

    std::string file_string = "";

    if (libs_file.is_open())
    {
        std::string line;
        std::vector<std::string> lines;

        int i = 0;
        while(std::getline(libs_file, line))
        {
            file_string = file_string + line;
            file_string = file_string + '\n';
            i+=1;
            if (i>17)
                break;
        }
            
            // lines.push_back(line);

        libs_file.close();

        // for (const std::string& fileLine : lines)
        //     std::cout << fileLine << std::endl;
    }
    return file_string;
}



int main() {

    Parse_Libs();

    std::string all_libs = Get_Base_Lib();

    // std::cout << all_libs << ".\n";


    std::string root = "lib_parser/parsed_libs";
    std::vector<fs::path> files = glob_cpp(root, ".txt");

    std::ifstream file;
    std::string line;
    for (auto &parsed_file : files)
    {
        std::string lib_string = "";
        file.open(parsed_file);

        std::getline(file, line); // consume date

        while(std::getline(file, line))
            all_libs = all_libs + line + "\n";
        file.close();


        // std::cout << "Add file " << parsed_file << " to llvm libs.\n";
    }


    all_libs = all_libs + "\n}";

    // std::cout << all_libs << ".\n";



    
    std::ofstream llvm_lib_file("src/libs_llvm/libs.cpp");
    llvm_lib_file << all_libs;
    llvm_lib_file.close();


    return 0;
}