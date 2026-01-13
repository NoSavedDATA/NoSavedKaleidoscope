#pragma once

#include "llvm/IR/Value.h"
#include "llvm/IR/Instructions.h"
#include <vector>
#include <map>

#include "any_map.h"
#include "array.h"
#include "bool.h"
#include "codegen_notes.h"
#include "data_tree.h"
#include "dict.h"
#include "float.h"
#include "float_vec.h"
#include "int.h"
#include "int_vec.h"
#include "list.h"
#include "map.h"
#include "nsk_vector.h"
#include "nullptr.h"
#include "str.h"
#include "str_vec.h"

using namespace llvm;

extern std::map<std::string, std::vector<char *>> ClassStrVecs;
extern std::map<std::string, DT_float_vec *> ClassFloatVecs;
extern std::map<std::string, float> NamedClassValues;
extern std::map<std::string, int> NamedInts;
extern std::map<std::string, DT_int_vec *> NamedIntVecs;
extern std::map<std::string, std::string> NamedObjects;
extern std::map<std::string, std::vector<std::pair<std::string, std::string>>> ScopeVarsToClean;
extern std::map<std::string, char *> ScopeNamesToClean;
extern std::map<int, std::map<std::string, std::vector<std::string>>> ThreadedScopeTensorsToClean;


extern std::map<std::string, std::vector<float>>  FloatVecAuxHash;

extern std::map<std::string, std::string> objectVecs;
extern std::map<std::string, int> objectVecsLastId;


enum Types {
  type_float = 0,
  type_tensor = 1,
  type_pinned_tensor = 2,
  type_object = 3,
  type_string = 4,
};

enum NameSolverTypes {
  type_self = 0,
  type_attr = 1,
  type_vec = 2,
  type_var = 3,
  type_object_name = 4,
  type_object_vec = 5
};
