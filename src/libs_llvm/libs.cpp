#include "llvm/IR/Value.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"

#include <iostream>
#include <memory>

using namespace llvm;

#include "../compiler_frontend/modules.h"

void Generate_LLVM_Functions() {
    PointerType *floatPtrTy, *int8PtrTy;

    floatPtrTy = Type::getFloatTy(*TheContext)->getPointerTo();
    int8PtrTy = Type::getInt8Ty(*TheContext)->getPointerTo();
    

	FunctionType *tuple_NewTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		true //vararg
	);
	TheModule->getOrInsertFunction("tuple_New", tuple_NewTy);

	FunctionType *tuple_StoreTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tuple_Store", tuple_StoreTy);

	FunctionType *tuple_LoadTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tuple_Load", tuple_LoadTy);

	FunctionType *tuple_printTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tuple_print", tuple_printTy);

	FunctionType *tuple_testTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tuple_test", tuple_testTy);

	FunctionType *tuple_CreateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tuple_Create", tuple_CreateTy);

	FunctionType *tuple_MarkToSweepTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tuple_MarkToSweep", tuple_MarkToSweepTy);

	FunctionType *tuple_IdxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("tuple_Idx", tuple_IdxTy);

	FunctionType *AnyVector_IdxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("AnyVector_Idx", AnyVector_IdxTy);

	FunctionType *CreateNotesVectorTy= FunctionType::get(
		int8PtrTy,
		{},
		false
	);
	TheModule->getOrInsertFunction("CreateNotesVector", CreateNotesVectorTy);

	FunctionType *Dispose_NotesVectorTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("Dispose_NotesVector", Dispose_NotesVectorTy);

	FunctionType *Add_Float_To_NotesVectorTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("Add_Float_To_NotesVector", Add_Float_To_NotesVectorTy);

	FunctionType *Add_String_To_NotesVectorTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("Add_String_To_NotesVector", Add_String_To_NotesVectorTy);

	FunctionType *str_CreateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_Create", str_CreateTy);

	FunctionType *str_LoadTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_Load", str_LoadTy);

	FunctionType *str_StoreTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_Store", str_StoreTy);

	FunctionType *str_MarkToSweepTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_MarkToSweep", str_MarkToSweepTy);

	FunctionType *str_CopyTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_Copy", str_CopyTy);

	FunctionType *str_str_addTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_str_add", str_str_addTy);

	FunctionType *str_float_addTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_float_add", str_float_addTy);

	FunctionType *float_str_addTy= FunctionType::get(
		int8PtrTy,
		{Type::getFloatTy(*TheContext), int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_str_add", float_str_addTy);

	FunctionType *PrintStrTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("PrintStr", PrintStrTy);

	FunctionType *split_str_to_floatTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("split_str_to_float", split_str_to_floatTy);

	FunctionType *cat_str_floatTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("cat_str_float", cat_str_floatTy);

	FunctionType *SplitStringTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("SplitString", SplitStringTy);

	FunctionType *str_split_idxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("str_split_idx", str_split_idxTy);

	FunctionType *str_to_floatTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_to_float", str_to_floatTy);

	FunctionType *StrToFloatTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("StrToFloat", StrToFloatTy);

	FunctionType *str_DeleteTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_Delete", str_DeleteTy);

	FunctionType *pinned_tensor_CreateTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("pinned_tensor_Create", pinned_tensor_CreateTy);

	FunctionType *pinned_tensor_Store_IdxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("pinned_tensor_Store_Idx", pinned_tensor_Store_IdxTy);

	FunctionType *pinned_tensor_CalculateIdxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("pinned_tensor_CalculateIdx", pinned_tensor_CalculateIdxTy);

	FunctionType *InstantiateObjectTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("InstantiateObject", InstantiateObjectTy);

	FunctionType *objHashTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("objHash", objHashTy);

	FunctionType *LoadObjectTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("LoadObject", LoadObjectTy);

	FunctionType *InitObjectVecWithNullTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("InitObjectVecWithNull", InitObjectVecWithNullTy);

	FunctionType *is_nullTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("is_null", is_nullTy);

	FunctionType *objAttr_var_from_varTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("objAttr_var_from_var", objAttr_var_from_varTy);

	FunctionType *objAttr_var_from_vecTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("objAttr_var_from_vec", objAttr_var_from_vecTy);

	FunctionType *objAttr_vec_from_varTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("objAttr_vec_from_var", objAttr_vec_from_varTy);

	FunctionType *objAttr_vec_from_vecTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("objAttr_vec_from_vec", objAttr_vec_from_vecTy);

	FunctionType *appendTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("append", appendTy);

	FunctionType *LoadObjectScopeNameTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("LoadObjectScopeName", LoadObjectScopeNameTy);

	FunctionType *PrintFloatTy= FunctionType::get(
		int8PtrTy,
		{Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("PrintFloat", PrintFloatTy);

	FunctionType *UnbugFloatTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("UnbugFloat", UnbugFloatTy);

	FunctionType *print_floatTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("print_float", print_floatTy);

	FunctionType *float_CreateTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_Create", float_CreateTy);

	FunctionType *float_LoadTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_Load", float_LoadTy);

	FunctionType *float_StoreTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_Store", float_StoreTy);

	FunctionType *float_MarkToSweepTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("float_MarkToSweep", float_MarkToSweepTy);

	FunctionType *StoreOnDemandNoFreeTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("StoreOnDemandNoFree", StoreOnDemandNoFreeTy);

	FunctionType *LoadOnDemandNoFreeTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("LoadOnDemandNoFree", LoadOnDemandNoFreeTy);

	FunctionType *nullptr_getTy= FunctionType::get(
		int8PtrTy,
		{},
		false
	);
	TheModule->getOrInsertFunction("nullptr_get", nullptr_getTy);

	FunctionType *str_vec_CreateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_vec_Create", str_vec_CreateTy);

	FunctionType *str_vec_LoadTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_vec_Load", str_vec_LoadTy);

	FunctionType *str_vec_StoreTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_vec_Store", str_vec_StoreTy);

	FunctionType *str_vec_MarkToSweepTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_vec_MarkToSweep", str_vec_MarkToSweepTy);

	FunctionType *PrintStrVecTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("PrintStrVec", PrintStrVecTy);

	FunctionType *LenStrVecTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("LenStrVec", LenStrVecTy);

	FunctionType *ShuffleStrVecTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("ShuffleStrVec", ShuffleStrVecTy);

	FunctionType *_glob_b_Ty= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("_glob_b_", _glob_b_Ty);

	FunctionType *IndexStrVecTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("IndexStrVec", IndexStrVecTy);

	FunctionType *str_vec_IdxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("str_vec_Idx", str_vec_IdxTy);

	FunctionType *str_vec_CalculateIdxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("str_vec_CalculateIdx", str_vec_CalculateIdxTy);

	FunctionType *str_vec_printTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_vec_print", str_vec_printTy);

	FunctionType *dictionary_CreateTy= FunctionType::get(
		int8PtrTy,
		{},
		false
	);
	TheModule->getOrInsertFunction("dictionary_Create", dictionary_CreateTy);

	FunctionType *dictionary_DisposeTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("dictionary_Dispose", dictionary_DisposeTy);

	FunctionType *float_vec_CreateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_vec_Create", float_vec_CreateTy);

	FunctionType *float_vec_LoadTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_vec_Load", float_vec_LoadTy);

	FunctionType *float_vec_StoreTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_vec_Store", float_vec_StoreTy);

	FunctionType *float_vec_MarkToSweepTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_vec_MarkToSweep", float_vec_MarkToSweepTy);

	FunctionType *float_vec_Store_IdxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_vec_Store_Idx", float_vec_Store_IdxTy);

	FunctionType *PrintFloatVecTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("PrintFloatVec", PrintFloatVecTy);

	FunctionType *zeros_vecTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("zeros_vec", zeros_vecTy);

	FunctionType *ones_vecTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("ones_vec", ones_vecTy);

	FunctionType *float_vec_IdxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("float_vec_Idx", float_vec_IdxTy);

	FunctionType *float_vec_CalculateIdxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("float_vec_CalculateIdx", float_vec_CalculateIdxTy);

	FunctionType *float_vec_first_nonzeroTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_vec_first_nonzero", float_vec_first_nonzeroTy);

	FunctionType *float_vec_printTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_vec_print", float_vec_printTy);

	FunctionType *tensor_CreateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tensor_Create", tensor_CreateTy);

	FunctionType *tensor_LoadTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tensor_Load", tensor_LoadTy);

	FunctionType *tensor_StoreTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tensor_Store", tensor_StoreTy);

	FunctionType *tensor_MarkToSweepTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tensor_MarkToSweep", tensor_MarkToSweepTy);

	FunctionType *gpuTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("gpu", gpuTy);

	FunctionType *tensor_gpuwTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("tensor_gpuw", tensor_gpuwTy);

	FunctionType *cpuTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("cpu", cpuTy);

	FunctionType *cpu_idxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("cpu_idx", cpu_idxTy);

	FunctionType *randu_likeTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("randu_like", randu_likeTy);

	FunctionType *write_zeroswTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("write_zerosw", write_zeroswTy);

	FunctionType *tensor_viewTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("tensor_view", tensor_viewTy);

	FunctionType *NewVecToTensorTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("NewVecToTensor", NewVecToTensorTy);

	FunctionType *tensor_CalculateIdxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("tensor_CalculateIdx", tensor_CalculateIdxTy);

}