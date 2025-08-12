#include <assert.h>
#include <glob.h>
#include <stdlib.h> 
#include <string.h>
#include <string>
#include <vector>

#include "../../../src/codegen/string.h"
#include "../../../src/data_types/str_vec.h"
#include "../../../src/nsk_cpp.h"



extern "C" void *_glob_b_(Scope_Struct *scope_struct, char *pattern) {
  glob_t glob_result;

  std::vector<char *> ret;

  if (glob(pattern, GLOB_TILDE, NULL, &glob_result) == 0) {
      for (size_t i = 0; i < glob_result.gl_pathc; ++i) {

        ret.push_back(strdup(glob_result.gl_pathv[i]));
      }
      globfree(&glob_result);
  }


  if (ret.size()<1)
    LogErrorC(scope_struct->code_line, "Glob failed to find files.");
    
  // Aux to not lose pointers
  std::string random_str = RandomString(15);
  StrVecAuxHash[random_str] = ret;
 
  return &StrVecAuxHash[random_str];
}