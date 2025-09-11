#include "llvm/IR/Value.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"


#include "../compiler_frontend/include.h"
#include "../compiler_frontend/modules.h"


std::map<std::string, StructType*> struct_types;

void Generate_Struct_Types() {
    llvm::Type *int8PtrTy = Type::getInt8Ty(*TheContext)->getPointerTo();
    llvm::Type *boolTy = Type::getInt1Ty(*TheContext);
    llvm::Type *floatTy = Type::getFloatTy(*TheContext);
    llvm::Type *longTy   = Type::getInt64Ty(*TheContext);
    llvm::Type *intTy = Type::getInt32Ty(*TheContext);
    llvm::Type *intPtrTy = Type::getInt32Ty(*TheContext)->getPointerTo();

    // std::vector<int>
    StructType *vecIntTy = StructType::create(
        *TheContext,
        {intPtrTy, longTy, longTy},
        "std.vector.int"
    );    
    StructType *int_vecTy  = StructType::create(
        *TheContext,
        {intTy, vecIntTy, intPtrTy},
        "DT_int_vec"
    );

    struct_types["int_vec"] = int_vecTy;
}