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

#include "../clean_up/clean_up.h"
#include "../common/extension_functions.h"
#include "../libs_llvm/so_libs.h"
#include "include.h"


std::map<std::string, std::string> lib_function_remaps;


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

    std::cout << ")\n\n";
}

void LibFunction::Link_to_LLVM(void *func_ptr) {

    llvm::Type *int8PtrTy = Type::getInt8Ty(*TheContext)->getPointerTo();
    llvm::Type *floatTy = Type::getFloatTy(*TheContext);
    llvm::Type *intTy = Type::getInt32Ty(*TheContext);

    llvm::Type *fn_return_type;
    std::vector<llvm::Type *> arg_types;

    std::string fn_return_type_str;
    std::vector<std::string> arg_types_str;


    
    if(ReturnType=="int"&&!IsPointer)
    {
        fn_return_type = intTy;
        fn_return_type_str = "int";
    }
    else if(ReturnType=="float"&&!IsPointer)
    {
        fn_return_type = floatTy;
        fn_return_type_str = "float";
    }
    else
    {
        fn_return_type = int8PtrTy;
        fn_return_type_str = "void_ptr";
    }

    for(int i=0; i<ArgNames.size(); ++i) {
        if(ArgNames[i]=="int"&&!ArgIsPointer[i])
        {
            arg_types.push_back(intTy);
            arg_types_str.push_back("int");
        }
        else if(ArgNames[i]=="float"&&!ArgIsPointer[i])
        {
            arg_types.push_back(floatTy);
            arg_types_str.push_back("float");
        }
        else
        {
            arg_types.push_back(int8PtrTy);
            arg_types_str.push_back("void_ptr");
        }
    }



    Lib_Functions_Return[Name] = fn_return_type_str;
    Lib_Functions_Args[Name] = std::move(arg_types_str);


    FunctionType *llvm_function = FunctionType::get(
        fn_return_type,
        arg_types,
        false
    );

    Function* funcDecl = cast<Function>(
        TheModule->getOrInsertFunction(Name, llvm_function).getCallee()
    );

    
    auto &JD = TheJIT->getMainJITDylib();
    
    SymbolMap symbols;
    symbols[TheJIT->Mangle(Name)] = {
        ExecutorAddr::fromPtr(func_ptr),
        JITSymbolFlags::Exported | JITSymbolFlags::Callable
    };

    if (auto Err = JD.define(absoluteSymbols(symbols)))
        LogError("Failed to define native function in JIT: " + toString(std::move(Err)));
}

void LibFunction::Add_to_Nsk_Dicts(void *func_ptr, std::string lib_name, bool is_default) {
    user_cpp_functions.push_back(Name);
    native_methods.push_back(Name);
    native_functions.push_back(Name);
    native_fn.push_back(Name);


    // Check if it is a data-type
    if(ReturnType!="float"||IsPointer)
    {
        functions_return_type[Name] = ReturnType;
        if(begins_with(ReturnType, "DT_"))
        {
            std::string nsk_data_type = ReturnType;
            nsk_data_type.erase(0, 3);

            if(!in_str(nsk_data_type, data_tokens))
                data_tokens.push_back(nsk_data_type);
        }
    }


    // Check for data-type operations return type
    for (std::string operation : op_map_names)
    {
        if (ends_with(Name, operation))
        {
            // std::cout << "OPERATION FUNCTION FOUND " << Name << ".\n\n\n\n\n";

            std::string operands = Name;

            size_t pos = operands.rfind(operation)-1;
            if (pos != std::string::npos) {
                operands.replace(pos, operation.length()+1, "");
            }


            std::string nsk_data_type = ReturnType;
            if(begins_with(nsk_data_type, "DT_"))
                nsk_data_type.erase(0, 3);

            ops_type_return.insert(std::make_pair(operands, nsk_data_type));

            break;
        }
    }
    

    // Check if it is a _Clean_Up or _backward function    
    if(ends_with(Name, "_Clean_Up"))
    {
        // std::cout << "FOUND CLEAN UP FUNCTION " << Name << ".\n";

        std::string nsk_type = Name;
        size_t pos = nsk_type.rfind("_Clean_Up");
        if (pos != std::string::npos) {
            nsk_type.replace(pos, 9, "");
        }

        using CleanupFunc = void(*)(void*);
        CleanupFunc casted_func_ptr = reinterpret_cast<CleanupFunc>(func_ptr);
        clean_up_functions[nsk_type] = casted_func_ptr;

    } else if (ends_with(Name, "_backward")) {
        // std::cout << "FOUND BACKWARD FN" << ".\n";
    }    
    else {

    }


    // Create a remap when using "import default"
    if(is_default && begins_with(Name, lib_name)) {
        std::string remaped_fn = erase_before_pattern(Name, "__");

        lib_function_remaps[remaped_fn] = Name;
    }
}



LibParser::LibParser(std::string lib_dir) {
    files = get_lib_files(lib_dir);
}


char LibParser::_getCh() {

    if(!file.is_open()) {
        file_idx++;
        file.open(files[file_idx]);
        // std::cout << "Now parsing" << files[file_idx] << ".\n";
    }

    if (file.get(ch)) {
        return ch;
    } else {
        // std::cout << "Reached file ending" << ".\n";
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
        LastChar = _getCh();
        while(LastChar!=10 && LastChar!=tok_eof && LastChar!=tok_finish)
            LastChar = _getCh();

        return tok_commentary;
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

    // std::cout << "\n\n";
}


void LibParser::ParseLibs() {
    token=0;

    // std::cout << "Begin parsing" << ".\n";

    token = _getToken();
    
    while(file_idx<files.size())
    {  
      if (token==tok_extern)
        ParseExtern();

      token = _getToken();
    }
    // std::cout << "\n\n";
}

void LibParser::PrintFunctions() {
    for (auto pair : Functions) {
        for (auto fn : pair.second)
            fn->Print();
    }
}


void LibParser::ImportLibs(std::string so_lib_path, std::string lib_name, bool is_default) {



    void* handle = dlopen(so_lib_path.c_str(), RTLD_LAZY);



    for (auto pair : Functions) { 
        for (auto fn : pair.second)
        {
            // std::cout << "Importing function:" << "\n";
            // fn->Print();
            

            void* func_ptr = dlsym(handle, fn->Name.c_str());
            if (!func_ptr) {
                LogError("Function " + fn->Name + " not found in library");
                continue;
            }

            fn->Link_to_LLVM(func_ptr);
            fn->Add_to_Nsk_Dicts(func_ptr, lib_name, is_default);
        }
    }
}