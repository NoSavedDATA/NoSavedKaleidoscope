#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <algorithm>
#include <glob.h>

#include "../common/include.h"
#include "../codegen/random.h"
#include "../codegen/string.h"
#include "../compiler_frontend/logging.h"
#include "../mangler/scope_struct.h"
#include "include.h"



extern "C" void *str_vec_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, void *init_val, DT_list *notes_vector)
{
  if (init_val!=nullptr)
  { 
    std::vector<char *> vec = *static_cast<std::vector<char *>*>(init_val);
    ClassStrVecs[name] = vec;
  }


  return nullptr;
}

extern "C" void *str_vec_Load(Scope_Struct *scope_struct, char *object_var_name) {
  // std::cout << "Load StrVec On Demand var to load: " << object_var_name << "\n";
  
  void *ret = &ClassStrVecs[object_var_name];
  return ret;
}


extern "C" void str_vec_Store(char *name, std::vector<char *> value, Scope_Struct *scope_struct){
  // std::cout << "STORING " << name << " on demand as StrVec type.\n";
  ClassStrVecs[name] = value;
  // move_to_char_pool(strlen(name)+1, name, "free");
  //delete[] name;
}


void str_vec_Clean_Up(void *data_ptr) {
  // ClassStrVecs.erase(name);
}




extern "C" float PrintStrVec(std::vector<char*> vec)
{
  for (int i=0; i<vec.size(); i++)
    std::cout << vec[i] << "\n";
  return 0;
}


extern "C" int LenStrVec(Scope_Struct *scope_struct, std::vector<char*> vec)
{
  return vec.size();
}


extern "C" void * ShuffleStrVec(Scope_Struct *scope_struct, std::vector<char*> vec)
{
  std::random_device rd;
  std::mt19937 g(rd()^get_millisecond_time());


  std::shuffle(vec.begin(), vec.end(), g);

  
  return &vec;
}



//deprecated
extern "C" char * shuffle_str(char *string_list)
{

  std::ostringstream oss;

  std::vector<std::string> splitted = split(string_list, "|||");


  std::random_shuffle(splitted.begin(), splitted.end());

  for (int i=0; i<splitted.size(); i++)
  {
    if (i>0)
      oss << "|||";
    oss << splitted[i];
  }

  std::string result = oss.str();

  char * cstr = new char [result.length()+1];
  std::strcpy (cstr, result.c_str());
    
  return cstr;
}


extern "C" void * _glob_b_(Scope_Struct *scope_struct, char *pattern) {
  glob_t glob_result;

  std::vector<char *> ret;

  if (glob(pattern, GLOB_TILDE, NULL, &glob_result) == 0) {
      for (size_t i = 0; i < glob_result.gl_pathc; ++i) {

        ret.push_back(strdup(glob_result.gl_pathv[i]));
      }
      globfree(&glob_result);
  }


  if (ret.size()<1)
    LogErrorS("Glob failed to find files.");
    
  // Aux to not lose pointers
  std::string random_str = RandomString(15);
  StrVecAuxHash[random_str] = ret;
  AuxRandomStrs[random_str] = "str_vec";
    
  return &StrVecAuxHash[random_str];
}


extern "C" char *IndexStrVec(std::vector<char*> vec, float _idx)
{

  int idx = (int) _idx;

  //std::cout << "Str vec indexed at [" << idx << "]: " << vec[idx] << "\n";
  
  
  return vec[idx];
}


extern "C" char * str_vec_Idx(Scope_Struct *scope_struct, char *vec_name, float _idx)
{

  // std::cout << "str_vec_Idx: " << vec_name << ".\n"; 
  // std::cout << "idx: " << _idx << ".\n";

  int idx = (int) _idx;

  std::vector<char*> vec = ClassStrVecs[vec_name];

  // std::cout << "Str Vec " << vec_name << "indexed at [" << idx << "]: " << vec[idx] << "\n";
  delete[] vec_name;

  return CopyString(vec[idx]);
}



extern "C" float str_vec_CalculateIdx(char *data_name, float first_idx, ...) {
  return first_idx;
}


extern "C" float str_vec_print(Scope_Struct *scope_struct, std::vector<char *> vec) {
  std::cout << "[";
  for (int i=0; i<vec.size()-1; i++)
    std::cout << vec[i] << ", ";

  std::cout << vec[vec.size()-1] << "]" << "\n";
  return 0;
}

