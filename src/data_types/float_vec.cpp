#include <iostream>
#include <vector>
#include <map>

#include "../codegen/random.h"
#include "include.h"

std::map<std::string, std::vector<float>> FloatVecAuxHash;



extern "C" float float_vec_Create(char *name, char *scopeless_name, float init_val, AnyVector *notes_vector, int thread_id, char *scope)
{

  delete[] name;
  delete[] scopeless_name;

  return 0;
}

extern "C" void *float_vec_Load(char *object_var_name, int thread_id) {
  std::cout << "Load StrVec On Demand var to load: " << object_var_name << "\n";
  
  void *ret = &ClassFloatVecs[object_var_name];
  delete[] object_var_name;
  return ret;
}

extern "C" float StoreFloatVecOnDemand(char *name, std::vector<float> value){
  std::cout << "STORING " << name << " on demand as float vec type" << ".\n";

  ClassFloatVecs[name] = value;
  move_to_char_pool(strlen(name)+1, name, "free");
  //delete[] name;
  return 0;
}

extern "C" float StoreFloatVecOnDemandOnIdx(char *name, float idx, float value){
  //std::cout << "STORING " << self << "." << object_var_name << " on demand as float vec type" << ".\n";

  ClassFloatVecs[name][(int)idx] = value;
  move_to_char_pool(strlen(name)+1, name, "free");
  //delete[] name;
  return 0;
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


extern "C" void * zeros_vec(float size) {
  // TODO: turn into python like expression [0]*size

  std::vector<float> vec = std::vector<float>(static_cast<size_t>(size), 0.0f);
  

  // Aux to not lose pointers
  std::string random_str = RandomString(15);
  FloatVecAuxHash[random_str] = vec;
  AuxRandomStrs[random_str] = "float_vec";
    
  return &FloatVecAuxHash[random_str];
}

extern "C" void * ones_vec(float size) {
  // TODO: turn into python like expression [0]*size

  std::vector<float> vec = std::vector<float>(static_cast<size_t>(size), 1.0f);
  

  // Aux to not lose pointers
  std::string random_str = RandomString(15);
  FloatVecAuxHash[random_str] = vec;
  AuxRandomStrs[random_str] = "float_vec";
    
  return &FloatVecAuxHash[random_str];
}