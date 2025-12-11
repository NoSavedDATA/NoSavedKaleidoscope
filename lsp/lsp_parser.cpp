#include <napi.h>
#include "json.hpp"

#include "../src/nsk_cpp.h"  // your C++ parser
#include "../src/include.h"  // your C++ parser
#include "../src/common/extension_functions.h"  // your C++ parser
#include "../src/compiler_frontend/include.h"  // your C++ parser

static void HandleImport() {
    Parser_Struct parser_struct;
    parser_struct.line = LineCounter;
    ParseImport(parser_struct);
}

static std::unique_ptr<ExprAST> HandleClass() {
    Parser_Struct parser_struct;
    parser_struct.line = LineCounter;
    return std::move(ParseClass(parser_struct));
}

static std::unique_ptr<FunctionAST> HandleDefinition() {
    std::cout << "--> Handle definition" << ".\n";    
    
    Parser_Struct parser_struct;
    if (std::unique_ptr<FunctionAST> FnAST = ParseDefinition(parser_struct)) {
        // FunctionProtos[FnAST->getProto().getName()] =
        //   std::make_unique<PrototypeAST>(FnAST->getProto());
        std::cout << "RETURN FROM HANDLE DEFINITION" << ".\n";
        return std::move(FnAST);
    } else {
        std::cout << "proto parse failed, return" << ".\n";
        // Skip token for error recovery.
        // getNextToken();
        return nullptr;
    }
}


struct all_expr {
    std::vector<std::unique_ptr<ExprAST>> exprs;
    std::vector<std::unique_ptr<FunctionAST>> exprs_fn;

    all_expr(std::vector<std::unique_ptr<ExprAST>> exprs,
             std::vector<std::unique_ptr<FunctionAST>> exprs_fn)
             : exprs(std::move(exprs)), exprs_fn(std::move(exprs_fn)) {}

};

static all_expr *MainLoop() {
    std::vector<std::unique_ptr<ExprAST>> exprs;
    std::vector<std::unique_ptr<FunctionAST>> exprs_fn;
    std::cout << "Main loop" << ".\n";
    std::cout << "curtok: " << CurTok << "/" << ReverseToken(CurTok) << ".\n";
    all_expr *all_expressions;
    while(true) {
        switch (CurTok) { 
        case tok_eof:
            all_expressions = new all_expr(std::move(exprs), std::move(exprs_fn));
            return all_expressions;
        case ';': // ignore top-level semicolons.
            getNextToken();
            break;
        case tok_space:
            getNextToken();
            break;
        case tok_tab:
            getNextToken();
            break;
        case tok_def: {
            std::unique_ptr<FunctionAST> fn_expr = HandleDefinition();
            if(fn_expr!=nullptr)
                exprs_fn.push_back(std::move(fn_expr));
            break;
        }
        case tok_class:
            exprs.push_back(HandleClass());
            break;
        case tok_import:
            HandleImport();
            break;
        default: 
            std::cout << "---curtok: " << CurTok << "/" << ReverseToken(CurTok) << ".\n";
            all_expressions = new all_expr(std::move(exprs), std::move(exprs_fn));
            return all_expressions;
        }
   }
}


Napi::Value parse(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    std::string code = info[0].As<Napi::String>();

    auto stream = std::make_unique<std::istringstream>(code);
    tokenizer.current = stream.get();
    tokenizer.inputStack.push(std::move(stream));

    tokenizer.current_dir = "./";
    tokenizer.current_file = "main";
    LineCounter = 1;

    std::cout << "Code is:\n" << code << ".\n";

    getNextToken();
    all_expr *all_expressions = MainLoop();
    

    std::vector<std::string> lines;

    nlohmann::json root;    // or whatever JSON lib you use
    root["lines"] = nlohmann::json::array();

    std::cout << "debug: ULULULU" << "\n";
    std::fflush(stdout);



    for (auto& expr : all_expressions->exprs_fn)
        root["lines"].push_back(expr->toJSON());
    

    for (auto& expr : all_expressions->exprs) {

        root["lines"].push_back(expr->toJSON());
        // auto ast_line = MyParser::parse_line(line); // your existing API
        // root["lines"].push_back(ast_line.toJson());
    }

    return Napi::String::New(env, root.dump());
}

// Module initialization
Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // InitializeNativeTarget();
    // InitializeNativeTargetAsmPrinter(); // Prepare for target hardware
    // InitializeNativeTargetAsmParser();
    set_functions_return_type();
    set_functions_args_type();
    set_user_functions();
    // TheJIT = ExitOnErr(KaleidoscopeJIT::Create());
    // InitializeModule();
    exports.Set("parse", Napi::Function::New(env, parse));
    return exports;
}

NODE_API_MODULE(lsp_parser, Init)
