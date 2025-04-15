#include <iostream>
#include <map>
#include <string>

#include "../common/extension_functions.h"
#include "../codegen/random.h"
#include "../codegen/string.h"
#include "../compiler_frontend/logging.h"
#include "../mangler/scope_struct.h"
#include "include.h"

std::map<std::string, std::string> AuxRandomStrs;



  
extern "C" float str_Create(char *name, char *scopeless_name, char *init_val, AnyVector *notes_vector, Scope_Struct *scope_struct) {

  // std::cout << "Creating string"  << ".\n";
  // std::cout << "Val: " << init_val << ".\n";
  
  pthread_mutex_lock(&clean_scope_mutex);
  NamedStrs[name] = init_val;
  //std::cout << "Store " << value << " at " << name << "\n";
  pthread_mutex_unlock(&clean_scope_mutex);
  move_to_char_pool(strlen(name)+1, name, "free");
  move_to_char_pool(strlen(name)+1, scopeless_name, "free");
  //delete[] name;

  return 0;
}

extern "C" void *str_Load(char *name, Scope_Struct *scope_struct){
  // std::cout << "Load str " << name << ".\n";
  //char *ret = CopyString(NamedStrs[name]);
  
  pthread_mutex_lock(&clean_scope_mutex);
  char *ret = NamedStrs[name];
  pthread_mutex_unlock(&clean_scope_mutex);
  move_to_char_pool(strlen(name)+1, name, "free");
  //delete[] name;

  return ret;
}


extern "C" float str_Store(char *name, char *value, Scope_Struct *scope_struct){
  
  //NamedStrs[name] = CopyString(value); //TODO: Break?
  
  pthread_mutex_lock(&clean_scope_mutex);
  NamedStrs[name] = value;
  //std::cout << "Store " << value << " at " << name << "\n";
  pthread_mutex_unlock(&clean_scope_mutex);
  move_to_char_pool(strlen(name)+1, name, "free");
  //delete[] name;

  return 0;
}





extern "C" float PrintStr(char* value){
  std::cout << "Str: " << value << "\n";
  return 0;
}


extern "C" float *split_str_to_float(char *in_string, int gather_position)
{
  std::vector<std::string> splitted = split_str(in_string, '/');

  float * ret = new float[1];

  if(gather_position<0)
    gather_position = splitted.size()+gather_position;

  ret[0] = std::stof(splitted[gather_position]);

  return ret;
}




extern "C" void *cat_str_float(char *c, float v)
{

  std::string s = c;
  std::string tmp = std::to_string((int)v);

  s = s + c;

  return str_to_char(s);
}







extern "C" void *SplitString(char *self, char *pattern)
{

  //std::cout << "\n\nSPLITTING: " << self << ", with pattern: " << pattern << "\n";


  std::vector<char *> result;
  char *input = strdup(self); // Duplicate the input string to avoid modifying the original
  char *token = strtok(input, pattern); // Get the first token

  while (token != nullptr) {
    result.push_back(token);
    token = strtok(nullptr, pattern); // Get the next token
  }

  std::string random_str = RandomString(15);
  StrVecAuxHash[random_str] = result;
  AuxRandomStrs[random_str] = "str_vec";
    
  return &StrVecAuxHash[random_str];
    
}




// INDEX METHODS

extern "C" char *SplitStringIndexate(char *name, char *pattern, float idx)
{
  pthread_mutex_lock(&clean_scope_mutex);
  char *self = NamedStrs[name];
  pthread_mutex_unlock(&clean_scope_mutex);
  //std::cout << "splitting: " << self << ", with pattern: " << pattern << "\n";

  
  std::vector<char *> splits;
  char *input = (char*)malloc(strlen(self) + 1);
  memcpy(input, self, strlen(self) + 1);
  //strcpy(input, self);

  char *saveptr;
  char *token = strtok_r(input, pattern, &saveptr); // Get the first token

  while (token != nullptr) {
    splits.push_back(token);
    token = strtok_r(nullptr, pattern, &saveptr); // Get the next token
  }


  //std::cout << "splitting " << name << "\n";

  if(splits.size()<=1)
  {
    std::string _err = "\nFailed to split.";
    LogErrorS(_err);
    std::cout << "" << name << "\n";
    return nullptr;
  }

  if (idx < 0)
    idx = splits.size() + idx;
  
  //std::cout << "Spltting " << self << " with " << pattern <<" at ["<<idx<<"]:  " << splits[idx] << "\n";
 
  // Convert the retained token to a std::string
  char *result = splits[idx];

  delete[] name;
  delete[] input;

  return result;
}


extern "C" float StrToFloat(char *in_str)
{
  //std::cout << "\n\nstr to float of " << in_str << "\n\n\n";

  char *copied = (char*)malloc(strlen(in_str) + 1);
  strcpy(copied, in_str);
  char *end;

  float ret = std::strtof(copied, &end);
  delete[] copied;
  return ret;
}