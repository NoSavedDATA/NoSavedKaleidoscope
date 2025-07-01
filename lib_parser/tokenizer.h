#pragma once



//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//






// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known words.
enum Token {
    tok_extern = -1,
    tok_C = -2,
    tok_any_ptr = -4,
    tok_identifier = -5,
    tok_three_dots = -6,
    tok_non_idented_identifier = -7,
    tok_commentary = -8,


    tok_space=10,
    tok_tab=9,

    tok_eof=4,
    tok_finish=127
};

  
std::string ReverseToken(int _char);
static int get_token();

extern std::map<int, std::string> token_to_string;

extern std::vector<char> ops;
extern std::vector<char> terminal_tokens; 


extern std::string IdentifierStr; // Filled in if tok_identifier
extern std::string Line;// Filled in if tok_identifier
extern float NumVal;             // Filled in if tok_number


extern int CurTok;
extern std::string FileRead;


extern int LineCounter;

extern int SeenTabs;
extern int LastSeenTabs;




int tokenize();
bool getDotToken();
void getNextToken();