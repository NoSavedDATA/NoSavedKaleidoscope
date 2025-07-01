#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>


// #include "libs_parser.h"
// #include "logging.h"
// #include "modules.h"
// #include "tokenizer.h"

#include "include.h"



using namespace llvm;
namespace fs = std::filesystem;

std::vector<fs::path> glob_cpp(const fs::path& rootDir, std::string extension=".cpp") {
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

inline std::vector<fs::path> get_lib_files(std::string lib_dir)
{
  std::vector<fs::path> files = glob_cpp(lib_dir);
  std::vector<fs::path> cu_files = glob_cpp(lib_dir, ".cu");

  files.insert(files.end(), cu_files.begin(), cu_files.end());

  return files;
}



LibFunction::LibFunction(std::string ReturnType, bool IsPointer, std::string Name, std::vector<std::string> ArgNames, std::vector<int> ArgIsPointer)
    : ReturnType(ReturnType), IsPointer(IsPointer), Name(Name), ArgNames(ArgNames), ArgIsPointer(ArgIsPointer) {}

void LibFunction::Print() {
    std::cout << "extern \"C\" " << ReturnType << " ";
    if(IsPointer)
      std::cout << "*";
    std::cout << Name << "(";

    for(int i=0;i<ArgNames.size(); ++i)
    {
      std::cout << ArgNames[i];
      if (ArgIsPointer[i])
        std::cout << " *"; 

      if(i<ArgNames.size()-1)
      std::cout << ", ";
    }

    std::cout << ")\n";
}



LibParser::LibParser(std::string lib_dir) {
    files = get_lib_files(lib_dir);
}


char LibParser::_getCh() {

    if(!file.is_open()) {
        file_idx++;
        file.open(files[file_idx]);
        std::cout << "Now parsing" << files[file_idx] << ".\n";
    }

    if (file.get(ch)) {
        return ch;
    } else {
        std::cout << "Reached file ending" << ".\n";
        file.close();

        file_idx++;
        if (file_idx<files.size())
        file.open(files[file_idx]);

        return tok_eof;
    }
}


int LibParser::_getTok() {

    while (LastChar==32 || LastChar==tok_tab)
    LastChar = _getCh();

    if(LastChar=='/')
    {
    LastChar = _getCh();
    if(LastChar=='/')
    {
        while(LastChar!=tok_space&&LastChar!=tok_eof&&LastChar!=tok_finish)
        LastChar = _getCh();
    } else
        return LastChar;
    }


    if(LastChar=='('||LastChar==')'||LastChar=='*'||LastChar==',')
    {
    char ret_char = LastChar;
    LastChar = _getCh();
    return ret_char;
    }


    if (LastChar=='"') {
    
    LastChar = _getCh();
    running_string = "";

    while(LastChar!='"')
    {
        running_string += LastChar;
        LastChar = _getCh();
    }
    LastChar = _getCh();

    return tok_str;
    }

    
    if (isalpha(LastChar)||LastChar=='_') {
    running_string = "";

    while(isalpha(LastChar)||LastChar=='_')
    {
        running_string += LastChar;
        LastChar = _getCh();
    }

    if(running_string=="extern")
        return tok_extern;


    return tok_identifier;
    }


    while (LastChar==tok_space||LastChar==tok_tab||LastChar==32) // skip blanks
    LastChar = _getCh();
    

    LastChar = _getCh();

    if(LastChar==tok_eof||LastChar==tok_finish)
    return tok_eof;

    return LastChar;
}

int LibParser::_getToken() {
    token = _getTok();
    return token;
}


void LibParser::ParseExtern() {

    std::string file_name = files[file_idx];

    _getToken(); // eat extern




    if (token!=tok_str)
      return;
    _getToken(); // eat "C"


    std::string return_type = running_string;
    _getToken(); // eat return type




    bool is_pointer = false;
    if (token=='*')
    {
      _getToken();
      is_pointer = true;
    }


    // std::cout << "\nextern \"C\" " << return_type << " " << running_string << ".\n";
    // std::cout << "" << ReverseToken(token) << ".\n";
    
    std::string fn_name = running_string;

    _getToken(); // eat fn name





    std::vector<std::string> arg_types;
    std::vector<int> arg_is_pointer;

    if(token!='(')
      return;
    // std::cout << "--------------Got parenthesis (" << ".\n";


    _getToken(); 

    while(token!=')')
    {
      arg_types.push_back(running_string);      
      _getToken(); // eat arg type

      if (token=='*')
      {
        _getToken();
        arg_is_pointer.push_back(1);
      } else
        arg_is_pointer.push_back(0);

      _getToken(); // eat arg name


      if(token==',')
        _getToken();

    }

    LibFunction *lib_fn = new LibFunction(return_type, is_pointer, fn_name, arg_types, arg_is_pointer);
    Functions[file_name].push_back(lib_fn);


    std::cout << "\n\n";

}

void LibParser::ParseLibs() {
    token=0;

    std::cout << "Begin parsing" << ".\n";

    token = _getToken();
    
    while(file_idx<files.size())
    {  
      if (token==tok_extern)
        ParseExtern();

      token = _getToken();
    }
    std::cout << "\n\n";
}

void LibParser::PrintFunctions() {
    for (auto pair : Functions) {
        for (auto fn : pair.second)
            fn->Print();
    }
}


void LibParser::ImportLibs(std::string so_lib_path) {
    std::cout << "IMPORTING LIB: " << so_lib_path << ".\n\n";
    void* handle = dlopen(so_lib_path.c_str(), RTLD_LAZY);


    if (!handle) {
        LogError("Lib " + so_lib_path + " not found.");
        return;
    }

    llvm::Type *int8PtrTy = Type::getInt8Ty(*TheContext)->getPointerTo();
    llvm::Type *floatTy = Type::getFloatTy(*TheContext);
    llvm::Type *intTy = Type::getInt32Ty(*TheContext);
    



    for (auto pair : Functions) {
        for (auto fn : pair.second)
        {
            std::cout << "Importing function:" << "\n";
            fn->Print();
            
            void* funcPtr = dlsym(handle, fn->Name.c_str());
            if (!funcPtr) {
                LogError("Function " + fn->Name + " not found in library");
                continue;
            }


            user_cpp_functions.push_back(fn->Name);
            native_methods.push_back(fn->Name);


            
            llvm::Type *fn_return_type;
            std::vector<llvm::Type *> arg_types;

            
            if(fn->ReturnType=="int"&&!fn->IsPointer)
                fn_return_type = intTy;
            else if(fn->ReturnType=="float"&&!fn->IsPointer)
                fn_return_type = floatTy;
            else
                fn_return_type = int8PtrTy;

            for(int i=0; i<fn->ArgNames.size(); ++i) {
                if(fn->ArgNames[i]=="int"&&!fn->ArgIsPointer[i])
                    arg_types.push_back(intTy);
                else if(fn->ArgNames[i]=="float"&&!fn->ArgIsPointer[i])
                    arg_types.push_back(floatTy);
                else
                    arg_types.push_back(int8PtrTy);
            }

            std::cout << "jitting with " << arg_types.size() << " args.\n";


            FunctionType *llvm_function = FunctionType::get(
                fn_return_type,
                arg_types,
                false
            );


            Function* funcDecl = cast<Function>(
                TheModule->getOrInsertFunction(fn->Name, llvm_function).getCallee()
            );

            
            auto &JD = TheJIT->getMainJITDylib();
            
            std::cout << "Adding symbol"  << ".\n";
            SymbolMap symbols;
            symbols[TheJIT->Mangle(fn->Name)] = {
                ExecutorAddr::fromPtr(funcPtr),
                JITSymbolFlags::Exported | JITSymbolFlags::Callable
            };
        
            if (auto Err = JD.define(absoluteSymbols(symbols))) {
                LogError("Failed to define native function in JIT: " + toString(std::move(Err)));
                continue;
            }

            


            std::cout << "\n";
        }
    }
}