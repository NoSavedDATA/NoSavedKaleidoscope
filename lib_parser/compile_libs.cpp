#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "include.h"

namespace fs = std::filesystem;

std::string Get_Base_Lib(std::string fpath, int until_line) {

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
            if (i>until_line)
                break;
        }
            
            // lines.push_back(line);

        libs_file.close();

        // for (const std::string& fileLine : lines)
        //     std::cout << fileLine << std::endl;
    }
    return file_string;
}





void CleanDeletedLibs() {

    std::string root = "lib_parser/parsed_libs";
    std::vector<fs::path> files = glob_cpp(root, "llvm_lib.txt");

    for (auto &file : files) {
        std::string src_file = Demangle_File_Name(file.string());

        if (!fexists(src_file+".cpp") && !fexists(src_file+".cu"))
        {

            std::string base_lib = file.string();
            const std::string suffix = "_llvm_lib.txt";
            if (base_lib.size() >= suffix.size() &&
                base_lib.compare(base_lib.size() - suffix.size(), suffix.size(), suffix) == 0) {
                base_lib.erase(base_lib.size() - suffix.size());
            }

            fremove(base_lib+"_llvm_lib.txt");
            fremove(base_lib+"_returns_dict.txt");
            fremove(base_lib+"_user_cpp.txt");
        }
    }
}


int main() {

    
    Parse_Libs();

    if (files.size()==0)
    {
        // std::cout << "FOUND NO FILE TO PARSE" << ".\n";
        return 0;
    }
    // std::cout << "files: " << files.size() << ".\n";

    CleanDeletedLibs();

    std::string all_libs = Get_Base_Lib("src/libs_llvm/libs.cpp", 17);
    std::string all_cpp = Get_Base_Lib("src/libs_llvm/user_cpp_functions.cpp", 17);
    std::string all_return_dicts = Get_Base_Lib("src/libs_llvm/functions_return.cpp", 23);
    std::string all_args_dicts = Get_Base_Lib("src/libs_llvm/functions_args.cpp", 13);
    std::string all_function_dicts = "";




    std::string root = "lib_parser/parsed_libs";
    std::vector<fs::path> files = glob_cpp(root, "llvm_lib.txt");
    std::vector<fs::path> cpp_files = glob_cpp(root, "user_cpp.txt");
    std::vector<fs::path> dict_files = glob_cpp(root, "returns_dict.txt");
    std::vector<fs::path> return_data_files = glob_cpp(root, "returns_data_dict.txt");
    std::vector<fs::path> args_dict_files = glob_cpp(root, "args_dict.txt");
    std::vector<fs::path> clean_up_files = glob_cpp(root, "clean_up.txt");
    std::vector<fs::path> backward_files = glob_cpp(root, "backward.txt");

    std::ifstream file;
    std::string line;
    
    for (auto &parsed_file : files)
    {
        file.open(parsed_file);

        std::getline(file, line); // consume date

        while(std::getline(file, line))
            all_libs = all_libs + line + "\n";
        file.close();
    }


    for (auto &parsed_file : cpp_files)
    {
        file.open(parsed_file);

        while(std::getline(file, line))
            all_cpp = all_cpp + "\t\t" + line + "\n";
        file.close();
    }

    for (auto &parsed_file : dict_files)
    {
        file.open(parsed_file);

        while(std::getline(file, line))
            all_return_dicts = all_return_dicts + "\t\t\t\t\t\t" + line + "\n";
        file.close();
    }
    
    all_return_dicts = all_return_dicts + "\n\t};\n\n";

    for (auto &parsed_file : return_data_files)
    {
        file.open(parsed_file);

        while(std::getline(file, line))
            all_return_dicts = all_return_dicts + line + "\n";
        file.close();
    }

    all_return_dicts = all_return_dicts + "\n}";



    for (auto &parsed_file : args_dict_files)
    {
        file.open(parsed_file);

        while(std::getline(file, line))
            all_args_dicts = all_args_dicts + "\t" + line + "\n";
        file.close();
    }

    for (auto &parsed_file : clean_up_files)
    {
        file.open(parsed_file);

        while(std::getline(file, line))
            all_function_dicts = all_function_dicts + line + "\n\n";
        file.close();
    }
    for (auto &parsed_file : backward_files)
    {
        file.open(parsed_file);
        while(std::getline(file, line))
            all_function_dicts = all_function_dicts + line + "\n\n";
        file.close();
    }

    all_libs = all_libs + "\n}";
    all_args_dicts = all_args_dicts + "\n}";

    
    all_cpp = all_cpp + "\n\t};\n\n\n";
    all_cpp = all_cpp + all_function_dicts;
    all_cpp = all_cpp + "\n}";


    
    // std::cout << all_libs << ".\n";
    // std::cout << all_cpp << ".\n";
    // std::cout << all_return_dicts << ".\n";



    Write_Txt("src/libs_llvm/functions_args.cpp", all_args_dicts);
    Write_Txt("src/libs_llvm/functions_return.cpp", all_return_dicts);
    Write_Txt("src/libs_llvm/libs.cpp", all_libs);
    Write_Txt("src/libs_llvm/user_cpp_functions.cpp", all_cpp);


    return 0;
}