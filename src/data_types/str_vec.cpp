#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <algorithm>
#include <glob.h>

#include "../common/include.h"
#include "../codegen/random.h"
#include "../compiler_frontend/logging.h"
#include "../mangler/scope_struct.h"
#include "include.h"



extern "C" float str_vec_Create(char *name, char *scopeless_name, float init_val, AnyVector *notes_vector, Scope_Struct *scope_struct)
{

  delete[] name;
  delete[] scopeless_name;

  return 0;
}

extern "C" void *str_vec_Load(char *object_var_name, Scope_Struct *scope_struct) {
  std::cout << "Load StrVec On Demand var to load: " << object_var_name << "\n";
  
  void *ret = &ClassStrVecs[object_var_name];
  delete[] object_var_name;
  return ret;
}


extern "C" void str_vec_Store(char *name, std::vector<char *> value, Scope_Struct *scope_struct){
  std::cout << "STORING " << name << " on demand as StrVec type.\n";
  ClassStrVecs[name] = value;
  move_to_char_pool(strlen(name)+1, name, "free");
  //delete[] name;
}




extern "C" float PrintStrVec(std::vector<char*> vec)
{
  for (int i=0; i<vec.size(); i++)
    std::cout << vec[i] << "\n";

  return 0;
}


extern "C" float LenStrVec(std::vector<char*> vec)
{
  return (float) vec.size();
}


extern "C" void * ShuffleStrVec(std::vector<char*> vec)
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


extern "C" void * _glob_b_(char *pattern) {
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