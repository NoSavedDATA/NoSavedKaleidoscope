#pragma once


#include "include.h"


class Expr {
    public:
     virtual ~Expr() = default;
     bool Vararg=false;

     virtual Lib_Info *Generate_LLVM(std::string, Lib_Info *) = 0;
};


class ExternFunctionExpr : public Expr {
    public:
        std::string ReturnType, FunctionName;
        std::vector<std::string> ArgTypes;


    ExternFunctionExpr(const std::string &ReturnType, const std::string &FunctionName, std::vector<std::string> ArgTypes, bool Vararg);

    Lib_Info *Generate_LLVM(std::string, Lib_Info *) override;
};


