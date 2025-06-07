#include <iostream>
#include <string>

#include "logging.h"

/// LogError* - These are little helper functions for error handling.
void LogErrorCuda(std::string Str) {

  if (Str!=" ")
    std::cout << "\nLine: " << -1 << "\n   \033[31m Error: \033[0m " << Str << "\n\n";
    
}