#include <algorithm>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <sstream>
#include <vector>

// #include "../common/include.h"
#include "../codegen/random.h"
#include "../codegen/string.h"
#include "../codegen/time.h"
#include "../compiler_frontend/logging.h"
#include "../mangler/scope_struct.h"
#include "include.h"


std::map<std::string, std::vector<char *>> StrVecAuxHash;



extern "C" void *str_vec_Create(Scope_Struct *scope_struct)
{
  return nullptr;
}



void str_vec_Clean_Up(void *data_ptr) {
}




extern "C" int LenStrVec(Scope_Struct *scope_struct, std::vector<char*> vec)
{
  return vec.size();
}


extern "C" std::vector<char*> *ShuffleStrVec(Scope_Struct *scope_struct, std::vector<char*> vec)
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


  std::shuffle(splitted.begin(), splitted.end(), MAIN_PRNG);

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




extern "C" char *IndexStrVec(std::vector<char*> vec, float _idx)
{
  int idx = (int) _idx;
  //std::cout << "Str vec indexed at [" << idx << "]: " << vec[idx] << "\n"; 
  return vec[idx];
}


extern "C" char * str_vec_Idx(Scope_Struct *scope_struct, std::vector<char*> vec, int idx)
{
  // std::cout << "str_vec_Idx: " << vec_name << ".\n"; 
  // std::cout << "idx: " << _idx << ".\n";
  // std::cout << "Str Vec " << vec_name << "indexed at [" << idx << "]: " << vec[idx] << "\n";
  return CopyString(scope_struct, vec[idx]);
}



extern "C" int str_vec_CalculateIdx(std::vector<char *> vec, int first_idx, ...) {
  if (first_idx<0)
    first_idx = vec.size()+first_idx;
  return first_idx;
}


extern "C" float str_vec_print(Scope_Struct *scope_struct, std::vector<char *> vec) {

  std::cout << "[";

  for (int i=0; i<vec.size()-1; i++)
    std::cout << vec[i] << ", ";

  std::cout << vec[vec.size()-1] << "]" << "\n";
  return 0;
}

