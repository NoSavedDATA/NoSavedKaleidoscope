
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
  { tok_non_idented_identifier, "non idented identifier" },
  // { tok_float, "float" },
  { tok_any_ptr, "any_pointer *" },
  { tok_three_dots, "..." },
  { tok_eof, "EOF" },
  { tok_finish, "FINISH" },
};



std::string IdentifierStr = ""; // Filled in if tok_identifier
std::string Line = ""; // Filled in if tok_identifier
std::string FileRead=""; // Filled in if tok_identifier
float NumVal;             // Filled in if tok_number




int LineCounter = 1;

int SeenTabs = 0;
int LastSeenTabs = 0;





std::string ReverseToken(int tok)
{
  // std::cout << "curtok: " << tok << ".\n";
  if (tok==tok_identifier)
    return IdentifierStr;
  if (token_to_string.count(tok)>0)
    return token_to_string[tok];
  return std::to_string(static_cast<char>(tok));
}

/// get_token - Return the next token from standard input.

std::vector<char> reset_chars = {tok_tab, ',', '(', ')', '{', '}'};




std::map<std::string, int> token_map = {
  {"extern", tok_extern},
  {"\"C\"", tok_C},
  // {"float", tok_float},
};

bool begin_with_space=false;
int return_string_token() { 
  if(token_map.count(IdentifierStr)>0)
  {
    return token_map[IdentifierStr];
  }
  if(begin_with_space)
    return tok_identifier;
  begin_with_space=true;
  return tok_non_idented_identifier;
}

char LastChar = ' ';
bool getDotToken() {

  int i=0;
  while(CurTok=='.')
  {
    getNextToken();
    i+=1;
  }

  // std::cout << "GOT DOT " << i << " Token " << ReverseToken(CurTok) << ".\n";
  if(i==3)
    return true;

  return false;
}


int tokenize() {

  // if (LastChar==tok_space||LastChar==tok_tab)
  // {
    while (LastChar==tok_space||LastChar==tok_tab)
    {
      if (LastChar==tok_space)
        begin_with_space=false;
      if (LastChar==tok_tab)
        begin_with_space=true;
      Line="";
      IdentifierStr="";
      LastChar=get_file_char();
      FileRead = FileRead+"\n";
    }
  //   return tok_space;
  // }


  while (LastChar==32) //blank space
  {
    begin_with_space=true;
    LastChar = get_file_char();
    Line += " ";
    FileRead = FileRead+" ";
  }

  if(LastChar=='/')
  {
    LastChar = get_file_char();
    if(LastChar=='/')
    {
      LastChar = get_file_char();
      begin_with_space=false;
      while(LastChar!=tok_space&&LastChar!=tok_eof&&LastChar!=tok_finish&&LastChar!=',')
        LastChar = get_file_char();
      return tok_commentary;
    } else
      return LastChar;
  }
  if (LastChar=='#')
  {
    begin_with_space=false;
    while(LastChar!=tok_space&&LastChar!=tok_eof&&LastChar!=tok_finish&&LastChar!=',')
      LastChar = get_file_char();
    return tok_commentary;
  }
  if (LastChar=='}') {
    begin_with_space=false;
    while(LastChar!=tok_space&&LastChar!=tok_eof&&LastChar!=tok_finish)
      LastChar = get_file_char();
    return tok_commentary;
  }
 

  if (isalpha(LastChar) || LastChar=='_' || LastChar=='\"') { // identifier: [a-zA-Z][a-zA-Z0-9]*

    IdentifierStr=LastChar;

    while (isalnum(LastChar) || LastChar=='_' || LastChar=='\"' || LastChar==':' || LastChar=='<' || LastChar=='>' || LastChar=='*'||LastChar==32) {
      
      LastChar = get_file_char();
      bool terminate=false;
      while(LastChar==32)
      {
        LastChar = get_file_char();
        terminate=true;
      }

      if(terminate)
      {
        if(LastChar!='*'&&LastChar!='>')
          return return_string_token();
        else {
          if(LastChar=='*')
          {
            Line += LastChar;
            IdentifierStr += LastChar;
            LastChar = get_file_char();
            if(LastChar!='>')
              return return_string_token();
          }
        }
          terminate = false;
      }

      if (isalnum(LastChar) || LastChar=='_' || LastChar=='\"' || LastChar==':' || LastChar=='<' || LastChar=='>' || LastChar=='*')
      {
        Line += LastChar;
        IdentifierStr += LastChar;
      } else
          return return_string_token();

    }
  }
 
  while (LastChar==32) //blank space
  {
    begin_with_space=true;
    LastChar = get_file_char();
    Line += " ";
    FileRead = FileRead+" ";
  }


  int ReturnChar = LastChar;
  LastChar = get_file_char();

  if(ReturnChar==tok_finish)
    return tok_finish;
  if(ReturnChar==tok_eof)
    return tok_eof;
  

  return ReturnChar;
}


int CurTok;
void getNextToken() {
  CurTok = tokenize();
}
