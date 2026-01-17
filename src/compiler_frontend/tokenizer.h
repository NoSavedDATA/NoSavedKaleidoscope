#pragma once


#include <iostream>
#include <filesystem>
#include <fstream>
#include <stack>
#include <string>
#include <map>
#include <memory>
#include <vector>
#include <unordered_map>






//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//


// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known words.
enum Token {
    tok_eof = -1,
    tok_finish = -52,

    // functions/classes
    tok_def = -2,
    tok_class = -77,
    tok_self = -78,
    tok_class_attr = -79,
    tok_extern = -3,
    tok_import = -51,


    // primary
    tok_identifier = -4,
    tok_number = -5,
    tok_int = -13,
    tok_str = -40, // ""
    tok_bool = -58,


    // bools
    tok_and = -59,
    tok_not = -60,
    tok_or = -61,
    tok_xor = -62,
    
    // control
    tok_main = -59,
    tok_if = -6,
    tok_then = -7,
    tok_else = -8,
    tok_for = -9,
    tok_while = -10,
    tok_spawn = -55,
    tok_channel = -56,
    tok_async = -22,
    tok_asyncs = -30,
    tok_async_finish = -23,
    tok_lock = -26,
    tok_unlock = -27,
    tok_tab = 9,
    tok_return = -32,
    tok_ret = -41,
    tok_as = -33,
    tok_in = -42,
 

    // operators
    tok_binary = -11,
    tok_unary = -12,
    tok_equal = -28,
    tok_diff = -34,
    tok_int_div = -77,
    tok_higher_eq = -35,
    tok_minor_eq = -36,
    tok_mod = -29,
    tok_arrow = -57,


    tok_space = -14,
    tok_commentary = -53,
    tok_lib_dt = -65,


    // var definition
    tok_attr_var = -18,
    tok_attr_tensor = -19,
    tok_post_class_attr_attr = -38,
    tok_post_class_attr_identifier = -39,

    tok_global = -49,
    tok_no_grad = -50,

    tok_data = -44,
    tok_struct = -54,
    tok_var = -45,
    tok_tuple = -46,
    tok_list = -47,
	tok_array = -64,
    tok_map = -48,
};


struct Tokenizer {
    std::stack<std::unique_ptr<std::istream>> inputStack;
    std::stack<std::string> dirs, files;
    std::stack<int> line_counters;
    std::istream* current;
    std::string current_dir = std::filesystem::current_path().string();
    char cur_c=' ';
    bool has_main = false, has_lib_file = false;
    std::ifstream lib_file;

    std::string token;
    std::string current_file = "main";
    
    public:
        Tokenizer();


        char get();
        std::istream& get_word();
        bool openFile(std::string);
        bool importFile(std::string, int);
};
  
std::string ReverseToken(int _char);
static int get_token(bool);




extern Tokenizer tokenizer;
extern std::string cur_line;


extern std::vector<std::string> LLVM_IR_Functions;
extern std::map<int, std::string> token_to_string;

extern std::vector<char> ops;
extern std::vector<char> terminal_tokens; 


extern std::string IdentifierStr; // Filled in if tok_identifier
extern float NumVal;             // Filled in if tok_number
extern bool BoolVal;


extern int LineCounter;

extern int SeenTabs;
extern int LastSeenTabs;

extern std::vector<std::string> data_tokens, compound_tokens, primary_data_tokens;


/// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
/// token the parser is looking at.  getNextToken reads another token from the
/// lexer and updates CurTok with its results.
extern int CurTok;
int getNextToken(bool block=false);

/// BinopPrecedence - This holds the precedence for each binary operator that is
/// defined.
extern std::map<char, int> BinopPrecedence;

/// get_tokenPrecedence - Get the precedence of the pending binary operator token.
int get_tokenPrecedence();


void get_tok_until_space();
// void get_tok_util_dot_or_space();
