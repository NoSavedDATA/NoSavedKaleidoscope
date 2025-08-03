#pragma once

#include <string>
#include <map>
#include <vector>


#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

struct LibFunction {
  std::string ReturnType;
  std::string Name;
  bool IsPointer;
  std::vector<std::string> ArgTypes;
  std::vector<std::string> ArgNames;
  std::vector<int> ArgIsPointer;

  LibFunction(std::string ReturnType, bool IsPointer, std::string Name,
              std::vector<std::string>, std::vector<std::string>, std::vector<int> ArgIsPointer);

  void Link_to_LLVM(void *);
  void Add_to_Nsk_Dicts(void *, std::string, bool);

  void Print(); 
};


struct LibParser {
  int file_idx=-1;
  std::string running_string="";
  int token;
  char LastChar = ' ';

  char ch;
  std::ifstream file;
  std::vector<fs::path> files;
  std::vector<std::string> function_names;
  std::vector<std::string> Initialize_Functions;
  
  std::map<std::string, std::vector<LibFunction*>> Functions;

  LibParser(std::string lib_dir); 


  char _getCh(); 
  int _getTok(); 
  int _getToken(); 

  void ParseExtern(); 

  void ParseLibs(); 

  void PrintFunctions(); 
  void ImportLibs(std::string, std::string, bool); 
};