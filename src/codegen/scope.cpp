#include <string>


#include "../backprop/include.h"
#include "../char_pool/include.h"
#include "../common/cu_commons.h"
#include "../cuda_threads/include.h"
#include "../compiler_frontend/include.h"
#include "../tensor/tensor_struct.h"



extern "C" char * ConcatScopeStr(char *lc, char *rc)
{
  // std::cout << "ConcatScopeStr: " << lc << " and " << rc << ".\n";
  std::string lstr = lc;


  if (in_str(lstr, globalVars))
  {

    size_t length_lc = strlen(lc) + 1;
    //char* result_cstr = new char[length_lc+length_rc];
    char *result_cstr = get_from_char_pool(length_lc, "concat free left");
    memcpy(result_cstr, lc, length_lc);
    move_to_char_pool(length_lc+1, lc, "concat free left");
    return result_cstr;
  }
  

  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  //char* result_cstr = new char[length_lc+length_rc];
  char *result_cstr = get_from_char_pool(length_lc+length_rc, "concat free left");
  
  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);


  move_to_char_pool(length_lc+1, lc, "concat free left");
  //delete[] lc;

  //std::cout << "ConcatScopeStr " << result_cstr << "\n";
  
  return result_cstr;
}


