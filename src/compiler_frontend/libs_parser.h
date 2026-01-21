#pragma once

#include <string>
#include <map>
#include <vector>

#include <filesystem>
#include <fstream>

#include "../data_types/data_tree.h"

namespace fs = std::filesystem;

struct LibFunction {
  std::string ReturnType;
  std::string Name;
  bool IsPointer, IsVarArg;
  std::vector<std::string> ArgTypes;
  std::vector<std::string> ArgNames;
  std::vector<int> ArgIsPointer;

  bool HasRetOverwrite;
  std::string LibType="";
  Data_Tree LibDT;

  int DefaultArgsCount;

  LibFunction(std::string ReturnType, bool IsPointer, std::string Name,
              std::vector<std::string>, std::vector<std::string>, std::vector<int> ArgIsPointer, bool,
              bool, std::string, Data_Tree, int);

  void Link_to_LLVM(void *, void *);
  void Add_to_Nsk_Dicts(void *, std::string, bool);

  void Print(); 
};


struct LibParser {
  int file_idx=-1, CurDefaultArgs=0;
  std::string running_string="";
  std::string lib_type="";
  std::string fn_name;
  Data_Tree lib_dt;

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

  void ParseDT(Data_Tree &);
  bool TryParseFnDataType();
  void ParseExtern(); 

  void ParseLibs(); 

  void PrintFunctions(); 
  void ImportLibs(std::string, std::string, bool); 
};
