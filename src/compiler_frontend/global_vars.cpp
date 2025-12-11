#include <cstddef>
#include <string>
#include <map>
#include <vector>


bool Shall_Exit = false;




// Tensor related
std::vector<std::string> return_tensor_functions, return_tensor_methods, return_tensor_fn, native_modules,
return_pinned_methods, vararg_methods, string_methods, native_methods, native_functions, native_fn,
return_string_fn, threaded_tensor_functions, require_scope_functions, notators_str;


std::map<std::string, std::string> reverse_ops;

std::vector<std::string> Sys_Arguments;


//global
std::map<std::string, std::string> floatFunctions;

bool has_main=false;

//global
std::vector<std::string> Classes;

std::map<size_t, std::vector<char *>> CharPool;

