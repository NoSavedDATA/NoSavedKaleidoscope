#pragma once



#include <iostream>
#include <map>
#include <string>
#include <vector>


#include "include.h"

int CurTok;


void Parse_Extern_Function() {
    std::cout << "Parsing extern function" << ".\n";
    std::cout << "Identifier: " << IdentifierStr << ".\n";

    CurTok = getNextToken();

    std::cout << "Post extern: " << ReverseToken(CurTok) << ".\n";

    if (CurTok!=tok_C)
        return;

}



void Parse_Primary() {

    CurTok = getNextToken();

    std::cout << "Parser got token " << ReverseToken(CurTok) << ".\n";

    switch (CurTok) {
        case tok_extern:
            return Parse_Extern_Function();
        case tok_eof:
        {
            std::cout << "EOF" << ".\n";
            std::exit(0);
        }
        case tok_finish:
        {
            std::cout << "FINISHING" << ".\n";
            std::exit(0);
        }
        default:
            return Parse_Primary();
    }
}


int main() {

    while (CurTok!=tok_finish&&CurTok!=tok_eof)
        Parse_Primary();

    std::cout << "Finishing"  << ".\n";
}