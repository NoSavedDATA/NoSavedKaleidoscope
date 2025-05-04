#pragma once


#include <algorithm>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <map>
#include <string>
#include <vector>


#include "include.h"

namespace fs = std::filesystem;


std::string file_name;



std::string Generate_LLVMs(std::string lib_string, std::vector<std::unique_ptr<Expr>> functions)
{
    for (auto &function : functions)
        lib_string = function->Generate_LLVM(file_name, lib_string);

   return lib_string; 
}



std::string Mangle_Lib_File_Name(std::string fname) {

    std::string path = fname;
    // Replace '/' with '_'
    std::replace(path.begin(), path.end(), '/', '_');

    // Replace ".cpp" with ".txt"
    size_t pos = path.rfind(".cpp");
    if (pos != std::string::npos) {
        path.replace(pos, 4, ".txt");
    }
    pos = path.rfind(".cu");
    if (pos != std::string::npos) {
        path.replace(pos, 4, ".txt");
    }
    
    path = "lib_parser/parsed_libs/" + path;
    return path;
}


void Write_Txt(std::string fname, std::string content) {

    std::ofstream lib_file(fname);

    lib_file << content;
    lib_file.close();
}

void Save_Lib_String(std::string lib_string) {

    fs::path file_path(file_name);

    auto ftime = fs::last_write_time(file_path);

    auto sctp = std::chrono::system_clock::time_point(ftime.time_since_epoch());

    std::time_t cftime = std::chrono::system_clock::to_time_t(sctp);

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&cftime), "%F %T");
    std::string string_last_modified = oss.str();

    // std::cout << "Last modified: " << string_last_modified << ".\n";


    std::string save_lib_string = string_last_modified + "\n" + lib_string;
    std::string lib_file_name = Mangle_Lib_File_Name(file_name);
    
    Write_Txt(lib_file_name, save_lib_string);

}



std::unique_ptr<Expr> Parse_Extern_Function() {

    std::string return_type;
    std::string function_name;
    std::vector<std::string> args_types;

    bool vararg=false;

    getNextToken();


    if (CurTok!=tok_C)
        return nullptr;
    

    
    getNextToken(); // function return

    if(CurTok==tok_float)
        return_type="float";
    else
        return_type=IdentifierStr;



    getNextToken(); 



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


    auto extern_fn = std::make_unique<ExternFunctionExpr>(return_type, function_name, std::move(args_types), vararg);

    return std::move(extern_fn);
}







std::vector<std::unique_ptr<Expr>> Parse_Primary(std::vector<std::unique_ptr<Expr>> functions) {

    getNextToken();

    // std::cout << "Parser got token " << ReverseToken(CurTok) << ".\n";

    switch (CurTok) {
        case tok_extern:
        {
            file_name = current_file_name;
            std::unique_ptr<Expr> extern_fn = Parse_Extern_Function();
            functions.push_back(std::move(extern_fn));
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
        std::string lib_string = "";
        std::vector<std::unique_ptr<Expr>> functions;

        while (CurTok!=tok_eof&&CurTok!=tok_finish)
            functions = Parse_Primary(std::move(functions));
        
        if(files.size()==0)
            break;

        lib_string = Generate_LLVMs(lib_string, std::move(functions));

        if (CurTok==tok_eof)
            getNextToken();

        Save_Lib_String(lib_string); 
    }
    
    // std::cout << "Finish parsing libraries"  << ".\n";
}