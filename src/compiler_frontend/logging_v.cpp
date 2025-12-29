#include <iostream>
#include <string>

#include "global_vars.h"
#include "tokenizer.h"


void LogErrorC(int line, std::string Str) {


  if(line!=-1)
  {
    if (tokenizer.current_file!="main file") {
      std::cout << "\n\n" << tokenizer.current_file << "\n";

      
      
      std::ifstream file;
      std::string str_line;
      
      int l=0;
      file.open(tokenizer.current_file);
      while(l<line&&std::getline(file, str_line))
        l++;

      file.close();
      printf("%s", str_line.c_str());
    }

    std::cout << "\nLine: " << line << "\n   ";
  } else
    std::cout << "\n\n" << tokenizer.current_file << "\n";


  std::cout << "\033[31m Error: \033[0m " << Str << "\n\n";  

  Shall_Exit = true;
}
