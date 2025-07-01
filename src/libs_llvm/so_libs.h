#pragma once

#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"


#include <map>
#include <string>
#include <vector>

using namespace llvm;




extern std::map<std::string, std::string> Lib_Functions_Return;
extern std::map<std::string, std::vector<std::string>> Lib_Functions_Args;


void Generate_Lib_Functions();