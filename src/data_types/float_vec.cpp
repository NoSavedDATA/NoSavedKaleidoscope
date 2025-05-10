#include <iostream>
#include <vector>
#include <map>

#include "../codegen/random.h"
#include "../mangler/scope_struct.h"
#include "include.h"


std::map<std::string, std::vector<float>> FloatVecAuxHash;



extern "C" std::vector<float> *float_vec_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, void *init_val, DT_list *notes_vector)
{
  // std::cout << "float_vec_Create" << ".\n";

  if (init_val!=nullptr)
  {
    std::vector<float> vec = *static_cast<std::vector<float>*>(init_val);
    ClassFloatVecs[name] = vec;
  }


  return nullptr;
}

extern "C" std::vector<float> *float_vec_Load(Scope_Struct *scope_struct, char *object_var_name) {
  // std::cout << "Load StrVec On Demand var to load: " << object_var_name << "\n";
  
  std::vector<float> *ret = &ClassFloatVecs[object_var_name];
  return ret;
}

extern "C" float float_vec_Store(char *name, std::vector<float> value, Scope_Struct *scope_struct){
  // std::cout << "STORING " << name << " on demand as float vec type" << ".\n";

  ClassFloatVecs[name] = value;
  return 0;
}

 
void float_vec_Clean_Up(void *data_ptr) {
  // ClassFloatVecs.erase(name);
}


extern "C" float float_vec_Store_Idx(char *name, float idx, float value, Scope_Struct *scope_struct){
  // std::cout << "float_vec_Store_Idx" << ".\n";
  //std::cout << "STORING " << self << "." << object_var_name << " on demand as float vec type" << ".\n";

  ClassFloatVecs[name][(int)idx] = value;
  return 0;
}



extern "C" void float_vec_Mark(std::vector<float> vec) {
  return;
}



extern "C" float PrintFloatVec(std::vector<float> vec)
{

  std::cout << "Float vector:\n[";
  for (int i=0; i<vec.size()-1; i++)
    std::cout << "" << vec[i] << ", ";
  std::cout << "" << vec[vec.size()-1];
  std::cout << "]\n\n";

  return 0;
}


extern "C" std::vector<float> * zeros_vec(Scope_Struct *scope_struct, float size) {
  // TODO: turn into python like expression [0]*size

  std::vector<float> vec = std::vector<float>(static_cast<size_t>(size), 0.0f);
  

  // Aux to not lose pointers
  std::string random_str = RandomString(15);
  FloatVecAuxHash[random_str] = vec;
  AuxRandomStrs[random_str] = "float_vec";
    
  return &FloatVecAuxHash[random_str];
}

extern "C" std::vector<float> * ones_vec(Scope_Struct *scope_struct, float size) {
  // TODO: turn into python like expression [0]*size

  std::vector<float> vec = std::vector<float>(static_cast<size_t>(size), 1.0f);
  

  // Aux to not lose pointers
  std::string random_str = RandomString(15);
  FloatVecAuxHash[random_str] = vec;
  AuxRandomStrs[random_str] = "float_vec";
    
  return &FloatVecAuxHash[random_str];
}


extern "C" float float_vec_Idx(Scope_Struct *scope_struct, char *vec_name, float _idx)
{
  int idx = (int) _idx;
  // std::cout << "float_vec_Idx on idx " << idx << " for the vector " << vec_name << ".\n";

  std::vector<float> vec = ClassFloatVecs[vec_name];
  // std::cout << "Loaded vec" << ".\n";
  float ret = vec[idx];
  // std::cout << "got: " << ret << ".\n";
  delete[] vec_name;
  // std::cout << "returning" << ".\n"; 
  return ret;
}



extern "C" float float_vec_CalculateIdx(char *data_name, float first_idx, ...) {
  return first_idx;
}




extern "C" float float_vec_first_nonzero(Scope_Struct *scope_struct, std::vector<float> vec)
{ 
  float idx = -1;
  for (int i=0; i<vec.size(); i++)
    if (vec[i]!=0)
    {
      idx = i;
      break;
    }

  return idx;
}



extern "C" float float_vec_print(Scope_Struct *scope_struct, std::vector<float> vec) {
  std::cout << "[";
  for (int i=0; i<vec.size()-1; i++)
    std::cout << vec[i] << ", ";

  std::cout << vec[vec.size()-1] << "]" << "\n";
  return 0;
}


