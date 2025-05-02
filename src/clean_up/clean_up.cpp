
#include <functional>
#include <map>
#include <string>

std::map<std::string, std::function<void(std::string, void *)>> clean_up_functions;