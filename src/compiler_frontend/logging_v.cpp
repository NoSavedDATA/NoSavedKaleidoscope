#include <iostream>
#include <string>


void LogErrorC(int line, std::string Str) {

  if(line!=-1)
    std::cout << "\nLine: " << line << "\n   ";   

  if (Str!=" ")
    std::cout << "\033[31m Error: \033[0m " << Str << "\n\n";  
}
