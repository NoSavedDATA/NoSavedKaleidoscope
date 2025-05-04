#pragma once




class Expr {
    public:
     virtual ~Expr() = default;
     bool Vararg=false;

     virtual std::string Generate_LLVM(std::string, std::string) = 0;
};


class ExternFunctionExpr : public Expr {
    public:
        std::string ReturnType, FunctionName;
        std::vector<std::string> ArgTypes;


    ExternFunctionExpr(const std::string &ReturnType, const std::string &FunctionName, std::vector<std::string> ArgTypes, bool Vararg);

    std::string Generate_LLVM(std::string, std::string) override;
};


