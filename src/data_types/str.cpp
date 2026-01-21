#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <sstream>

#include "../char_pool/include.h"
#include "../common/extension_functions.h"
#include "../codegen/random.h"
#include "../codegen/string.h"
#include "../compiler_frontend/logging_v.h"
#include "../compiler_frontend/tokenizer.h"
#include "../mangler/scope_struct.h"
#include "../pool/include.h" 
#include "include.h"


#ifdef _WIN32
#define strtok_r strtok_s
#endif




  
extern "C" char *str_Create(Scope_Struct *scope_struct) {
  return nullptr;
}




void str_Clean_Up(void *data_ptr)
{
  char *char_ptr = static_cast<char *>(data_ptr);
  move_to_char_pool(strlen(char_ptr)+1, char_ptr, "Mark sweep of str");
}




extern "C" char *str_Copy(Scope_Struct *scope_struct, char *str) {
  // std::cout << "Copying string: " << str << ".\n";
  char *ret = CopyString(scope_struct, str);
  return ret;
}


extern "C" char *str_CopyArg(Scope_Struct *scope_struct, char *str, char *argname) {
  // std::cout << "Copying string: " << str << ".\n";
  char *ret = CopyString(scope_struct, str);
  return ret;
}

extern "C" char * str_str_add(Scope_Struct *scope_struct, char *lc, char *rc)
{
  // std::cout << "Concat fn" << ".\n";
  // std::cout << "Concat: " << lc << " -- " << rc << ".\n";
  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  char *result_cstr = allocate<char>(scope_struct, length_lc+length_rc, "str");


  
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

  char *result_cstr = allocate<char>(scope_struct, length_lc+length_rc, "str");

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

  char *result_cstr = allocate<char>(scope_struct, length_lc+length_rc, "str");

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
  
  char *result_cstr = allocate<char>(scope_struct, length_lc+length_rc, "str");

  
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

  char *result_cstr = allocate<char>(scope_struct, length_lc+length_rc, "str");
  
  memcpy(result_cstr, lc_str.c_str(), length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);

  return result_cstr;
}



extern "C" char * str_bool_add(Scope_Struct *scope_struct, char *lc, bool rc)
{
  size_t length_lc = strlen(lc);

  size_t length_rc;
  std::string r;
  if(rc) {
    length_rc = 5;
    r = "true";
  } else {
    length_rc = 6;
    r = "false";
  }

  char *result_cstr = allocate<char>(scope_struct, length_lc+length_rc, "str");

  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, r.c_str(), length_rc);

  return result_cstr;
}

extern "C" char * bool_str_add(Scope_Struct *scope_struct, bool lc, char *rc)
{
  size_t length_lc;
  std::string l;
  if(rc) {
    length_lc = 4;
    l = "true";
  } else {
    length_lc = 5;
    l = "false";
  }

  size_t length_rc = strlen(rc)+1;

  char *result_cstr = allocate<char>(scope_struct, length_lc+length_rc, "str");

  memcpy(result_cstr, l.c_str(), length_lc);
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
    LogErrorC(scope_struct->code_line, _err);
    std::cout << "" << self << "\n";
    return nullptr;
  }

  if (idx < 0)
    idx = splits.size() + idx;
  
  // std::cout << "Spltting " << self << " with " << pattern <<" at ["<<idx<<"]:  " << splits[idx] << "\n";
 
  // Convert the retained token to a std::string
  char *result = CopyString(scope_struct, splits[idx]);


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



extern "C" bool str_str_different(Scope_Struct scope_struct, char *l, char *r) {
  return strcmp(l, r);
}
extern "C" bool str_str_equal(Scope_Struct scope_struct, char *l, char *r) {
  return !strcmp(l, r);
}



extern "C" void str_Delete(char *in_str) {
  // std::cout << "str_Delete of " << in_str << ".\n";
  move_to_char_pool(strlen(in_str)+1, in_str, "free");
}



extern "C" char *readline(Scope_Struct *scope_struct) {
    std::string line;
    if (!std::getline(std::cin, line)) {
        // EOF or error
        char* buf = (char*)malloc(1);
        buf[0] = '\0';
        return buf;
    }

    char* buf = (char*)malloc(line.size() + 1);
    if (!buf) return nullptr;
    std::memcpy(buf, line.c_str(), line.size() + 1);
    return buf;
}
