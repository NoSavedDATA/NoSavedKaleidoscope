#include "llvm/IR/Value.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"


#include "../compiler_frontend/include.h"
#include "../compiler_frontend/modules.h"


std::map<std::string, StructType*> struct_types;

void Generate_Struct_Types() {
    // Get llvm types
    llvm::Type *int8PtrTy = Type::getInt8Ty(*TheContext)->getPointerTo();
    llvm::Type *boolTy = Type::getInt1Ty(*TheContext);
    llvm::Type *floatTy = Type::getFloatTy(*TheContext);
    llvm::Type *longTy   = Type::getInt64Ty(*TheContext);
    llvm::Type *intTy = Type::getInt32Ty(*TheContext);
    llvm::Type *intPtrTy = Type::getInt32Ty(*TheContext)->getPointerTo();


    // std::vector<void*>
    StructType *void_vecTy = StructType::create(
        *TheContext,
        {int8PtrTy, longTy, longTy},
        "std.vector.int"
    );


    // --- DT_int_vecs ---
    // std::vector<int>
    StructType *vecIntTy = StructType::create(
        *TheContext,
        {intPtrTy, longTy, longTy},
        "std.vector.int"
    );
    // DT_int_vecs
    StructType *int_vecTy  = StructType::create(
        *TheContext,
        {intTy, vecIntTy, intPtrTy},
        "DT_int_vec"
    );
    struct_types["int_vec"] = int_vecTy;

    // array
    struct_types["vec"] = StructType::create(
        *TheContext,
        {intTy, intTy, intTy, int8PtrTy},
        "DT_array"
    ); 

    // --- map ---
    struct_types["map_node"]  = StructType::create(
        *TheContext,
        {int8PtrTy, int8PtrTy, int8PtrTy},
        "DT_map"
    );
    // map
    struct_types["map"]  = StructType::create(
        *TheContext,
        {intTy, intTy, intTy, intTy, intTy, struct_types["map_node"]->getPointerTo()->getPointerTo()},
        "DT_map"
    );
    
    // --- Scope_Struct --- 
    // GC
    StructType *GC_Struct_Type = StructType::create(
        *TheContext,
        {intTy, longTy, void_vecTy},
        "GC"
    );
    // Scope_Struct
    StructType *Scope_Struct_Type = StructType::create(
        *TheContext,
        // {intTy, intTy, GC_Struct_Type},
        {intTy, intTy, ArrayType::get(int8PtrTy, ContextStackSize), intTy, int8PtrTy, GC_Struct_Type},
        "GC"
    );
    struct_types["GC"] = GC_Struct_Type;
    struct_types["scope_struct"] = Scope_Struct_Type;
}
