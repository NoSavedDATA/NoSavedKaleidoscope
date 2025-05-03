
#include<map>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<ctype.h>
#include<string>
#include<iostream>

#include <algorithm>
#include <array>

#include "include.h"



//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//




bool in_char(char ch, const std::vector<char>& list) {
  // Use std::find to efficiently search the list for the character
  return std::find(list.begin(), list.end(), ch) != list.end();
}


// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known words.



std::map<int, std::string> token_to_string = {
  { tok_extern, "extern" },
  { tok_C, "C" },
  { tok_float, "float" },
  { tok_any_ptr, "any_pointer *" },
  { wait_token, "wait token" },
  { tok_finish, "FINISH" },
};



std::string IdentifierStr = ""; // Filled in if tok_identifier
std::string Line = ""; // Filled in if tok_identifier
float NumVal;             // Filled in if tok_number




int LineCounter = 1;

int SeenTabs = 0;
int LastSeenTabs = 0;





std::string ReverseToken(int _char)
{
  return token_to_string[_char];
}

/// get_token - Return the next token from standard input.

std::vector<char> reset_chars = {tok_tab, ',', '(', ')', '{', '}'};




std::map<std::string, int> token_map = {
  {"extern", tok_extern},
  {"\"C\"", tok_C},
};



int getNextToken() {

  char LastChar = get_file_char();

    if(LastChar==tok_finish||LastChar==tok_eof)
    {
      std::cout << "GOT FINISH" << ".\n";
    }

  while (LastChar==tok_space)
  {
    std::cout << "Line is:" << Line << "\n";
    Line="";
    IdentifierStr="";
    LastChar=get_file_char();
    if(LastChar==tok_finish||LastChar==tok_eof)
    {
      std::cout << "GOT FINISH" << ".\n";
    }
  }

    if(LastChar==tok_finish||LastChar==tok_eof)
    {
      std::cout << "GOT FINISH" << ".\n";
    }
  if (in_char(LastChar, reset_chars))
  {
    IdentifierStr = "";
    LastChar = get_file_char();
    if(LastChar==tok_finish||LastChar==tok_eof)
    {
      std::cout << "GOT FINISH" << ".\n";
    }
  }
    if(LastChar==tok_finish||LastChar==tok_eof)
    {
      std::cout << "GOT FINISH" << ".\n";
    }

  while (LastChar==32) //blank space
  {
    LastChar = get_file_char();
    if(LastChar==tok_finish||LastChar==tok_eof)
    {
      std::cout << "GOT FINISH" << ".\n";
    }
    Line += " ";
  }
  
  IdentifierStr="";
  if (isalpha(LastChar) || LastChar=='_' || LastChar=='\"') { // identifier: [a-zA-Z][a-zA-Z0-9]*
    while (isalpha(LastChar) || LastChar=='_' || LastChar=='\"') {
      IdentifierStr += LastChar;
      Line += LastChar;
      LastChar = get_file_char();
    if(LastChar==tok_finish||LastChar==tok_eof)
    {
      std::cout << "GOT FINISH" << ".\n";
    }
    }

    if(LastChar==tok_finish||LastChar==tok_eof)
    {
      std::cout << "GOT FINISH" << ".\n";
    }

    if(token_map.count(IdentifierStr)>0)
    {
      std::cout << "\nFound token, got line:\n" << Line << ".\n\n";
      return token_map[IdentifierStr];
    }
    if(LastChar==tok_finish||LastChar==tok_eof)
    {
      std::cout << "GOT FINISH" << ".\n";
    }
    return tok_identifier;
  }
 
    if(LastChar==tok_finish||LastChar==tok_eof)
    {
      std::cout << "GOT FINISH" << ".\n";
    }


  return wait_token;
}


