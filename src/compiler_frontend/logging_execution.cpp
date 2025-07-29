#include <iostream>
#include <string>



/// LogError* - These are little helper functions for error handling.
void LogErrorEE(int line, std::string Str) {
  if (Str!=" ")
    std::cout << "\nLine: " << line << "\n   \033[31m Error: \033[0m " << Str << "\n\n"; 
  std::exit(0);
}
