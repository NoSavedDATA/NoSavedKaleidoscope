#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <map>
#include <vector>

#include "../char_pool/include.h"
#include "../pool/include.h"
#include "random.h"





extern "C" char *GetEmptyChar(Scope_Struct *scope_struct)
{
  // char *empty_char = allocate<char>(scope_struct, 1, "str");
  char *empty_char = (char*)malloc(1);
  empty_char[0] = '\0';
  return empty_char;
}

extern "C" void FreeCharFromFunc(char *_char, char *func) {
  std::cout << "FREEING " << _char << " at function: " << func << "\n";
  free(_char);
  std::cout << "freed" << "\n";
}


extern "C" void FreeChar(char *_char) {
  // std::cout << "free" << ".\n";
  // std::cout << "FREEING " << _char << "\n";

  move_to_char_pool(strlen(_char)+1, _char, "free");
}




extern "C" char *CopyString(Scope_Struct *scope_struct, char *in_str)
{
  size_t length = strlen(in_str) + 1; // +1 for null terminator

  char *copied = allocate<char>(scope_struct, length, "str");
  // std::cout << "COPY STR GOT: " << copied << ".\n";
  // char *copied = (char*)malloc(length);
  memcpy(copied, in_str, length);

  return copied;
}

extern "C" char * ConcatStr(Scope_Struct *scope_struct, char *lc, char *rc)
{
  // std::cout << "Concat fn" << ".\n";
  // std::cout << "Concat: " << lc << " -- " << rc << ".\n";
  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator

  // char *result_cstr = allocate<char>(scope_struct, length_lc+length_rc, "str");
  char *result_cstr = (char*)malloc(length_lc+length_rc);
  
  
  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);

  //std::cout << "ConcatStr " << result_cstr << "\n";

  return result_cstr;
}

extern "C" char * ConcatStrFreeLeft(Scope_Struct *scope_struct, char *lc, char *rc)
{
  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  
  // char *result_cstr = allocate<char>(scope_struct, length_lc+length_rc, "str");
  char *result_cstr = (char*)malloc(length_lc+length_rc);
  
  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);

  // move_to_char_pool(length_lc+1, lc, "concat free left");
  
  return result_cstr;
}




extern "C" char * ConcatFloatToStr(Scope_Struct *scope_struct, char *lc, float r)
{
  std::string l = lc;
  int _r = r;

  std::string result_str = l + std::to_string(_r);
  // char *result_cstr = allocate<char>(scope_struct, result_str.length()+1, "str");
  char *result_cstr = (char*)malloc(result_str.length()+1);

  std::strcpy(result_cstr, result_str.c_str());

  return result_cstr;
}

extern "C" char * ConcatNumToStrFree(Scope_Struct *scope_struct, char *lc, float r)
{
  std::string l = lc;
  int _r = r;

  std::string result_str = l + std::to_string(_r);
  // char *result_cstr = allocate<char>(scope_struct, result_str.length()+1, "str");
  char *result_cstr = (char*)malloc(result_str.length()+1);
  std::strcpy(result_cstr, result_str.c_str());

  
  return result_cstr;
}
