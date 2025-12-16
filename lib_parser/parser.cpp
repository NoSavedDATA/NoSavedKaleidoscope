#pragma once


#include <algorithm>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <vector>


#include "include.h"

namespace fs = std::filesystem;


std::string file_name;



Lib_Info::Lib_Info() {};





Lib_Info *Generate_LLVMs(Lib_Info *lib_info, std::vector<std::unique_ptr<Expr>> functions)
{
    for (auto &function : functions)
        lib_info = function->Generate_LLVM(file_name, lib_info);

   return lib_info; 
}






void Write_Txt(std::string fname, std::string content) {
    std::ofstream lib_file(fname);

    lib_file << content;
    lib_file.close();
}
void Write_Append(std::string fname, std::string content) {
    std::ofstream lib_file(fname, std::ios::app);

    lib_file << content;
    lib_file.close();
}

void Save_llvm_string(Lib_Info *lib_info) {

    fs::path file_path(file_name);

    auto ftime = fs::last_write_time(file_path);

    auto sctp = std::chrono::system_clock::time_point(ftime.time_since_epoch());

    std::time_t cftime = std::chrono::system_clock::to_time_t(sctp);

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&cftime), "%F %T");
    std::string string_last_modified = oss.str();



    Lib_Files *lib_files = new Lib_Files();
    lib_files->Mangle_Lib_File_Name(file_name);


    std::string save_llvm_string = string_last_modified + "\n" + lib_info->llvm_string;
    
    Write_Txt(lib_files->file_name, save_llvm_string);
    Write_Txt(lib_files->user_cpp, lib_info->functions_string);
    Write_Txt(lib_files->returns_dict, lib_info->dict_string);
    Write_Txt(lib_files->returns_data_dict, lib_info->return_data_string);
    Write_Txt(lib_files->args_dict, lib_info->arg_types_string);
    Write_Txt(lib_files->clean_up, lib_info->clean_up_functions);
    Write_Txt(lib_files->backward, lib_info->backward_functions);

    free(lib_files);
    free(lib_info);
}


void ParseToNextLine() {
    while(CurTok!=tok_eof&&CurTok!=tok_space&&CurTok!=tok_finish)
        getNextToken();
    getNextToken();
}

std::unique_ptr<Expr> Parse_CPP_Function() {


    bool log = contains_str(file_name, "array");

    getNextToken();

    if(CurTok!=tok_identifier) {
        // ParseToNextLine();
        auto expr = std::make_unique<PlaceholderExpr>();
        return expr;
    }
    getNextToken();

    // if (log) {
    //     std::cout << "------" << Line << ".\n";
    // }
    
    if (CurTok!='(') {
        auto expr = std::make_unique<PlaceholderExpr>();
        return expr;
    }

    std::string function_name = IdentifierStr;
    getNextToken();


    auto expr = std::make_unique<CppFunctionExpr>(function_name);
    return expr;
}


std::unique_ptr<Expr> Parse_Extern_Function() {

    bool log = contains_str(file_name, "array");

    std::string return_type;
    std::string function_name;
    std::vector<std::string> args_types;

    bool vararg=false;

    getNextToken(); // eat extern


    if (CurTok!=tok_C)
    {
        auto expr = std::make_unique<PlaceholderExpr>();
        return expr;
    }
    

    
    getNextToken(); // eat "C"


    return_type=IdentifierStr;


    getNextToken(); // eat function return



    if (CurTok=='*')
    {
        return_type=return_type+"*";
        getNextToken();   
    }


    
    // function name
    if(CurTok!=tok_identifier)
    {
        std::cout << "INVALID TOKEN " << ReverseToken(CurTok) << " FOUND WHILE PARSING FUNCTION. EXPECTED THE FUNCTION NAME" << ".\n";
        std::cout << "" << IdentifierStr << ".\n";
        std::exit(0);
    }
    function_name = IdentifierStr;

    // if (log)
    //     std::cout << "Parsing: " << function_name << ".\n";
    



    getNextToken(); // eat '('
    
    if(CurTok!='(')
    {
        std::cout << "INVALID TOKEN " << ReverseToken(CurTok) << " FOUND WHILE PARSING FUNCTION. EXPECTED '('" << ".\n";
        std::exit(0);
    }


    getNextToken(); // get first argument type

    if (CurTok!=')')
    { 
        while(true) {
            std::string arg = IdentifierStr;
            if (CurTok=='.')
            {
                if (getDotToken())
                {
                    for (int i=0;i<10;i++)
                        args_types.push_back(args_types[args_types.size()-1]);
                    vararg=true;
                    // getNextToken();
                    break;
                }
            }
            getNextToken(); // eat type
            
            while(CurTok=='*'||CurTok=='&')
            {
                if(CurTok=='*')
                    arg+="*";
                if(CurTok=='&')
                    arg+="&";
                getNextToken(); // eat * or &
            }

            getNextToken(); // eat arg_name

            args_types.push_back(arg);
            if (CurTok==',')
                getNextToken();
            if (CurTok==')')
                break;

        }
    }
    getNextToken(); // eat ')'


    // std::cout << "Function:\n\tReturn Type:\t" << return_type << "\n\tName:\t\t" << function_name << "\n\tArgs:\t\t";

    auto extern_fn = std::make_unique<ExternFunctionExpr>(return_type, function_name, std::move(args_types), vararg);

    return std::move(extern_fn);
}







// todo: this function may break recursion stack
std::vector<std::unique_ptr<Expr>> Parse_Primary(std::vector<std::unique_ptr<Expr>> functions) {

    getNextToken();



    // bool log = contains_str(file_name, "array");
    // if (log) {
    //     // std::cout << "Parser got token " << ReverseToken(CurTok) << ".\n";
    //     // std::cout << "" << FileRead << "\n";
    // }


    switch (CurTok) {
        case tok_extern:
        {
            file_name = current_file_name;
            std::unique_ptr<Expr> extern_fn = Parse_Extern_Function();
            functions.push_back(std::move(extern_fn));
            return std::move(functions);
        }
        case tok_non_idented_identifier:
        {
            file_name = current_file_name;
            std::unique_ptr<Expr> cpp_fn = Parse_CPP_Function();
            functions.push_back(std::move(cpp_fn));
            return std::move(functions);
        }
        case tok_eof:
            return std::move(functions);
        case tok_finish:
            return std::move(functions);
        case tok_space:
        {
            while(CurTok==tok_space||CurTok==tok_tab)
                getNextToken();
            return std::move(functions);
        }
        default:
            return Parse_Primary(std::move(functions));
    }
}




void Parse_Libs() {


    while (CurTok!=tok_finish)
    {
        bool has_functions=true;
        Lib_Info *lib_info = new Lib_Info();

        std::vector<std::unique_ptr<Expr>> functions;
        
        while (CurTok!=tok_eof&&CurTok!=tok_finish)
            functions = Parse_Primary(std::move(functions));
        

        if (functions.size()>0)
            lib_info = Generate_LLVMs(lib_info, std::move(functions));
        else
            has_functions=false;

        if (CurTok==tok_eof)
            getNextToken();

        // std::cout << lib_info->dict_string << ".\n\n";
        // std::cout << "functions list: " << lib_info->functions_string << ".\n";
        // std::cout << "------------" << ".\n\n\n\n";

        if (has_functions)
            Save_llvm_string(lib_info);
        else
            free(lib_info);
    }
    
    // std::cout << "Finish parsing libraries"  << ".\n";
}
