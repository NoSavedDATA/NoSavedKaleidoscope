#pragma once


#include "include.h"


class Expr {
    public:
     virtual ~Expr() = default;
     bool Vararg=false;

     virtual Lib_Info *Generate_LLVM(std::string, Lib_Info *) = 0;
     virtual Lib_Info *Generate_Args_Dict(Lib_Info *) = 0;
};


class ExternFunctionExpr : public Expr {
    public:
        std::string ReturnType, FunctionName;
        std::vector<std::string> ArgTypes;


    ExternFunctionExpr(const std::string &ReturnType, const std::string &FunctionName, std::vector<std::string> ArgTypes, bool Vararg);

    Lib_Info *Generate_LLVM(std::string, Lib_Info *) override;
    Lib_Info *Generate_Args_Dict(Lib_Info *) override;
};



class PlaceholderExpr : public Expr {
    public:
    PlaceholderExpr();
    Lib_Info *Generate_LLVM(std::string, Lib_Info *) override;
    Lib_Info *Generate_Args_Dict(Lib_Info *) override;
};


class CppFunctionExpr : public Expr {
    public:
        std::string FunctionName;
    CppFunctionExpr(const std::string &);
    Lib_Info *Generate_LLVM(std::string, Lib_Info *) override;
    Lib_Info *Generate_Args_Dict(Lib_Info *) override;
};