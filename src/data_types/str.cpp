#include <iostream>
#include <map>
#include <string>

#include <sstream>
#include <iomanip>

#include "../common/extension_functions.h"
#include "../codegen/random.h"
#include "../codegen/string.h"
#include "../compiler_frontend/logging.h"
#include "../mangler/scope_struct.h"
#include "include.h"

std::map<std::string, std::string> AuxRandomStrs;



  
extern "C" char *str_Create(Scope_Struct *scope_struct, char *name, char *scopeless_name, char *init_val, DT_list *notes_vector) {

  // std::cout << "Creating string"  << ".\n";
  // std::cout << "Val: " << init_val << ".\n";
  
  NamedStrs[name] = init_val;
  //std::cout << "Store " << value << " at " << name << "\n";

  return init_val;
}

extern "C" char *str_Load(Scope_Struct *scope_struct, char *name){
  // std::cout << "Load str " << name << ".\n";
  //char *ret = CopyString(NamedStrs[name]);
  
  char *ret = NamedStrs[name];
  // move_to_char_pool(strlen(name)+1, name, "free");
  //delete[] name;

  return ret;
}



extern "C" float str_Store(char *name, char *value, Scope_Struct *scope_struct) {
  
  //NamedStrs[name] = CopyString(value); //TODO: Break?
  
  
  NamedStrs[name] = value;
  //std::cout << "Store " << value << " at " << name << "\n";
  
  // std::cout << "STORING STRING " << value << " AT " << name << ".\n";
  
  return 0;
}


void str_Clean_Up(void *data_ptr)
{
  // std::cout << "str_Clean_Up" << ".\n";
  char *char_ptr = static_cast<char *>(data_ptr);
  move_to_char_pool(strlen(char_ptr)+1, char_ptr, "Mark sweep of str");
}




extern "C" char *str_Copy(Scope_Struct *scope_struct, char *str) {
  // std::cout << "Copying string: " << str << ".\n";
  char *ret = CopyString(str);
  return ret;
}

extern "C" char * str_str_add(Scope_Struct *scope_struct, char *lc, char *rc)
{
  // std::cout << "Concat fn" << ".\n";
  // std::cout << "Concat: " << lc << " -- " << rc << ".\n";
  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  char *result_cstr = get_from_char_pool(length_lc+length_rc, "concat");
  //char* result_cstr = new char[length_lc+length_rc]; 
  
  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);

  //std::cout << "ConcatStr " << result_cstr << "\n";

  return result_cstr;
}



extern "C" char * str_int_add(Scope_Struct *scope_struct, char *lc, int rc)
{
  // std::cout << "Concat string and int fn" << ".\n";
  // std::cout << "Int is: " << rc << ".\n";
  // std::cout << "Concat: " << lc << " -- " << rc << ".\n";

  size_t length_lc = strlen(lc);

  // Convert the float to a string
  std::stringstream ss;
  ss << rc; // Adjust precision as needed
  std::string rc_str = ss.str();
  size_t length_rc = rc_str.length() + 1; // +1 for null terminator

  char *result_cstr = get_from_char_pool(length_lc + length_rc, "str_float_add");

  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc_str.c_str(), length_rc);

  // std::cout << "cat str int " << result_cstr << ".\n";

  return result_cstr;
}



extern "C" char * str_float_add(Scope_Struct *scope_struct, char *lc, float rc)
{
  // std::cout << "Concat string and float fn" << ".\n";
  // std::cout << "Concat: " << lc << " -- " << rc << ".\n";

  size_t length_lc = strlen(lc);

  // Convert the float to a string
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2) << rc; // Adjust precision as needed
  std::string rc_str = ss.str();
  size_t length_rc = rc_str.length() + 1; // +1 for null terminator

  char *result_cstr = get_from_char_pool(length_lc + length_rc, "str_float_add");

  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc_str.c_str(), length_rc);


  return result_cstr;
}

extern "C" char * int_str_add(Scope_Struct *scope_struct, int lc, char *rc)
{   
  std::stringstream ss;
  ss << lc; // Adjust precision as needed
  std::string lc_str = ss.str();
  size_t length_lc = lc_str.length(); // +1 for null terminator

  
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  

  char *result_cstr = get_from_char_pool(length_lc+length_rc, "float_str_add");

  
  memcpy(result_cstr, lc_str.c_str(), length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);


  return result_cstr;
}


extern "C" char * float_str_add(Scope_Struct *scope_struct, float lc, char *rc)
{  
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2) << lc; // Adjust precision as needed
  std::string lc_str = ss.str();
  size_t length_lc = lc_str.length();
 
  
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  

  char *result_cstr = get_from_char_pool(length_lc+length_rc, "float_str_add");

  
  memcpy(result_cstr, lc_str.c_str(), length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);


  return result_cstr;
}





extern "C" float PrintStr(char* value){
  std::cout << "Str: " << value << "\n";
  return 0;
}






extern "C" char *cat_str_float(char *c, float v)
{

  std::string s = c;
  std::string tmp = std::to_string((int)v);

  s = s + c;

  return str_to_char(s);
}







extern "C" std::vector<char *> *SplitString(Scope_Struct *scope_struct,char *self, char *pattern)
{

  std::cout << "\n\nSPLITTING: " << self << ", with pattern: " << pattern << "\n";


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

extern "C" char *str_split_idx(Scope_Struct *scope_struct, char *self, char *pattern, int idx)
{
  // std::cout << "SLIPPINT " << self << ".\n";
  
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


  if(splits.size()<=1)
  {
    std::string _err = "\nFailed to split.";
    LogErrorS(_err);
    std::cout << "" << self << "\n";
    return nullptr;
  }

  if (idx < 0)
    idx = splits.size() + idx;
  
  // std::cout << "Spltting " << self << " with " << pattern <<" at ["<<idx<<"]:  " << splits[idx] << "\n";
 
  // Convert the retained token to a std::string
  char *result = CopyString(splits[idx]);


  free(input);

  return result;
}


extern "C" float str_to_float(Scope_Struct *scope_struct, char *in_str)
{
  // std::cout << "Execution: str_to_float" << ".\n";
  // std::cout << "\n\nstr to float of " << in_str << "\n\n\n";

  char *copied = (char*)malloc(strlen(in_str) + 1);
  strcpy(copied, in_str);
  char *end;

  float ret = std::strtof(copied, &end);

  free(copied);
  // std::cout << "return from to float: " << ret << ".\n";

  return ret;
}


extern "C" float StrToFloat(Scope_Struct *scope_struct,char *in_str)
{
  // std::cout << "Execution: StrToFloat" << ".\n";
  // std::cout << "\n\nstr to float of " << in_str << "\n\n\n";

  char *copied = (char*)malloc(strlen(in_str) + 1);
  strcpy(copied, in_str);
  char *end;

  float ret = std::strtof(copied, &end);
  delete[] copied;
  return ret;
}



extern "C" void str_Delete(char *in_str) {
  // std::cout << "str_Delete of " << in_str << ".\n";
  move_to_char_pool(strlen(in_str)+1, in_str, "free");
}

