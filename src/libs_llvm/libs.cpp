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

	FunctionType *object_Attr_on_Offset_floatTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("object_Attr_on_Offset_float", object_Attr_on_Offset_floatTy);

	FunctionType *object_Attr_on_Offset_intTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("object_Attr_on_Offset_int", object_Attr_on_Offset_intTy);

	FunctionType *object_Attr_on_OffsetTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("object_Attr_on_Offset", object_Attr_on_OffsetTy);

	FunctionType *object_Load_on_Offset_floatTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("object_Load_on_Offset_float", object_Load_on_Offset_floatTy);

	FunctionType *object_Load_on_Offset_intTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("object_Load_on_Offset_int", object_Load_on_Offset_intTy);

	FunctionType *object_Load_on_OffsetTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("object_Load_on_Offset", object_Load_on_OffsetTy);

	FunctionType *object_ptr_Load_on_OffsetTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("object_ptr_Load_on_Offset", object_ptr_Load_on_OffsetTy);

	FunctionType *object_ptr_Attribute_objectTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("object_ptr_Attribute_object", object_ptr_Attribute_objectTy);

	FunctionType *reluTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("relu", reluTy);

	FunctionType *geluTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("gelu", geluTy);

	FunctionType *sigmoidTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("sigmoid", sigmoidTy);

	FunctionType *_tanhTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("_tanh", _tanhTy);

	FunctionType *softmaxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("softmax", softmaxTy);

	FunctionType *int_CreateTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_Create", int_CreateTy);

	FunctionType *int_LoadTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_Load", int_LoadTy);

	FunctionType *int_StoreTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_Store", int_StoreTy);

	FunctionType *btc_multTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("btc_mult", btc_multTy);

	FunctionType *btc_multTTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("btc_multT", btc_multTTy);

	FunctionType *PrintDimsTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("PrintDims", PrintDimsTy);

	FunctionType *StoreDimsOnDemandTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("StoreDimsOnDemand", StoreDimsOnDemandTy);

	FunctionType *CalculateIdxOffsetTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("CalculateIdxOffset", CalculateIdxOffsetTy);

	FunctionType *tensor_shapeTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tensor_shape", tensor_shapeTy);

	FunctionType *RandomCropTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("RandomCrop", RandomCropTy);

	FunctionType *RandomHorizontalFlipTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("RandomHorizontalFlip", RandomHorizontalFlipTy);

	FunctionType *NormalizeImgTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("NormalizeImg", NormalizeImgTy);

	FunctionType *JitterTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("Jitter", JitterTy);

	FunctionType *RandomStrOnDemandTy= FunctionType::get(
		int8PtrTy,
		{},
		false
	);
	TheModule->getOrInsertFunction("RandomStrOnDemand", RandomStrOnDemandTy);

	FunctionType *GetEmptyCharTy= FunctionType::get(
		int8PtrTy,
		{},
		false
	);
	TheModule->getOrInsertFunction("GetEmptyChar", GetEmptyCharTy);

	FunctionType *FreeCharFromFuncTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("FreeCharFromFunc", FreeCharFromFuncTy);

	FunctionType *FreeCharTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("FreeChar", FreeCharTy);

	FunctionType *CopyStringTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("CopyString", CopyStringTy);

	FunctionType *ConcatStrTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("ConcatStr", ConcatStrTy);

	FunctionType *ConcatStrFreeLeftTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("ConcatStrFreeLeft", ConcatStrFreeLeftTy);

	FunctionType *ConcatStrFreeRightTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("ConcatStrFreeRight", ConcatStrFreeRightTy);

	FunctionType *ConcatStrFreeTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("ConcatStrFree", ConcatStrFreeTy);

	FunctionType *ConcatFloatToStrTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("ConcatFloatToStr", ConcatFloatToStrTy);

	FunctionType *ConcatNumToStrFreeTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("ConcatNumToStrFree", ConcatNumToStrFreeTy);

	FunctionType *tensor_tensor_mmaTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tensor_tensor_mma", tensor_tensor_mmaTy);

	FunctionType *tensor_tensor_addTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tensor_tensor_add", tensor_tensor_addTy);

	FunctionType *tensor_tensor_subTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tensor_tensor_sub", tensor_tensor_subTy);

	FunctionType *tensor_tensor_equalTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tensor_tensor_equal", tensor_tensor_equalTy);

	FunctionType *tensor_tensor_multTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tensor_tensor_mult", tensor_tensor_multTy);

	FunctionType *tensor_tensor_divTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tensor_tensor_div", tensor_tensor_divTy);

	FunctionType *MarkToSweep_MarkTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("MarkToSweep_Mark", MarkToSweep_MarkTy);

	FunctionType *MarkToSweep_Unmark_ScopefulTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("MarkToSweep_Unmark_Scopeful", MarkToSweep_Unmark_ScopefulTy);

	FunctionType *MarkToSweep_Unmark_ScopelessTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("MarkToSweep_Unmark_Scopeless", MarkToSweep_Unmark_ScopelessTy);

	FunctionType *scope_struct_CreateTy= FunctionType::get(
		int8PtrTy,
		{},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Create", scope_struct_CreateTy);

	FunctionType *scope_struct_CopyTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Copy", scope_struct_CopyTy);

	FunctionType *scope_struct_OverwriteTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Overwrite", scope_struct_OverwriteTy);

	FunctionType *scope_struct_DiveTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Dive", scope_struct_DiveTy);

	FunctionType *set_scope_at_returnTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("set_scope_at_return", set_scope_at_returnTy);

	FunctionType *set_scope_not_at_returnTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("set_scope_not_at_return", set_scope_not_at_returnTy);

	FunctionType *set_scope_first_argTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("set_scope_first_arg", set_scope_first_argTy);

	FunctionType *set_scope_scopeTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("set_scope_scope", set_scope_scopeTy);

	FunctionType *set_scope_thread_idTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("set_scope_thread_id", set_scope_thread_idTy);

	FunctionType *set_scope_has_gradTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("set_scope_has_grad", set_scope_has_gradTy);

	FunctionType *set_scope_function_nameTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("set_scope_function_name", set_scope_function_nameTy);

	FunctionType *get_scope_first_argTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("get_scope_first_arg", get_scope_first_argTy);

	FunctionType *get_scope_scopeTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("get_scope_scope", get_scope_scopeTy);

	FunctionType *get_scope_thread_idTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("get_scope_thread_id", get_scope_thread_idTy);

	FunctionType *get_scope_has_gradTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("get_scope_has_grad", get_scope_has_gradTy);

	FunctionType *scope_struct_Reset_ThreadsTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Reset_Threads", scope_struct_Reset_ThreadsTy);

	FunctionType *scope_struct_Increment_ThreadTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Increment_Thread", scope_struct_Increment_ThreadTy);

	FunctionType *set_scope_objectTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("set_scope_object", set_scope_objectTy);

	FunctionType *get_scope_objectTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("get_scope_object", get_scope_objectTy);

	FunctionType *scope_struct_Save_for_AsyncTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Save_for_Async", scope_struct_Save_for_AsyncTy);

	FunctionType *scope_struct_Load_for_AsyncTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Load_for_Async", scope_struct_Load_for_AsyncTy);

	FunctionType *scope_struct_Store_Asyncs_CountTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Store_Asyncs_Count", scope_struct_Store_Asyncs_CountTy);

	FunctionType *scope_struct_PrintTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Print", scope_struct_PrintTy);

	FunctionType *scope_struct_Get_Async_ScopeTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Get_Async_Scope", scope_struct_Get_Async_ScopeTy);

	FunctionType *scope_struct_Alloc_MarkSweepMapTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Alloc_MarkSweepMap", scope_struct_Alloc_MarkSweepMapTy);

	FunctionType *scope_struct_Copy_MarkSweepMapTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Copy_MarkSweepMap", scope_struct_Copy_MarkSweepMapTy);

	FunctionType *scope_struct_SweepTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Sweep", scope_struct_SweepTy);

	FunctionType *scope_struct_Clean_ScopeTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Clean_Scope", scope_struct_Clean_ScopeTy);

	FunctionType *scope_struct_DeleteTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Delete", scope_struct_DeleteTy);

	FunctionType *network_emaTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("network_ema", network_emaTy);

	FunctionType *LSTMTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("LSTM", LSTMTy);

	FunctionType *LSTM_CreateTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("LSTM_Create", LSTM_CreateTy);

	FunctionType *backpropTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("backprop", backpropTy);

	FunctionType *load_imgTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("load_img", load_imgTy);

	FunctionType *gload_imgTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("gload_img", gload_imgTy);

	FunctionType *wload_imgTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("wload_img", wload_imgTy);

	FunctionType *wload_img_resizeTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("wload_img_resize", wload_img_resizeTy);

	FunctionType *load_preprocess_imgTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("load_preprocess_img", load_preprocess_imgTy);

	FunctionType *logETy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("logE", logETy);

	FunctionType *logE2Ty= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("logE2", logE2Ty);

	FunctionType *clipTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("clip", clipTy);

	FunctionType *AttrTensorOnIdxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("AttrTensorOnIdx", AttrTensorOnIdxTy);

	FunctionType *AttrTensorOnIdxTensorTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("AttrTensorOnIdxTensor", AttrTensorOnIdxTensorTy);

	FunctionType *AttrPinnedFromTensorOnIdxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("AttrPinnedFromTensorOnIdx", AttrPinnedFromTensorOnIdxTy);

	FunctionType *IdxTensorTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("IdxTensor", IdxTensorTy);

	FunctionType *IdxTensorWithTensorTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("IdxTensorWithTensor", IdxTensorWithTensorTy);

	FunctionType *_exitTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("_exit", _exitTy);

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
		{int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy},
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

	FunctionType *nullptr_getTy= FunctionType::get(
		int8PtrTy,
		{},
		false
	);
	TheModule->getOrInsertFunction("nullptr_get", nullptr_getTy);

	FunctionType *rl_discounted_returnTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("rl_discounted_return", rl_discounted_returnTy);

	FunctionType *printTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("print", printTy);

	FunctionType *PrintTensorTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("PrintTensor", PrintTensorTy);

	FunctionType *print_tensorTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("print_tensor", print_tensorTy);

	FunctionType *PrintTensorFTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("PrintTensorF", PrintTensorFTy);

	FunctionType *PrintTensorI8Ty= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("PrintTensorI8", PrintTensorI8Ty);

	FunctionType *ConcatScopeStrTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("ConcatScopeStr", ConcatScopeStrTy);

	FunctionType *tensor_transposeTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tensor_transpose", tensor_transposeTy);

	FunctionType *pinned_tensor_CreateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("pinned_tensor_Create", pinned_tensor_CreateTy);

	FunctionType *pinned_tensor_LoadTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("pinned_tensor_Load", pinned_tensor_LoadTy);

	FunctionType *pinned_tensor_Store_IdxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getFloatTy(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("pinned_tensor_Store_Idx", pinned_tensor_Store_IdxTy);

	FunctionType *pinned_tensor_CalculateIdxTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("pinned_tensor_CalculateIdx", pinned_tensor_CalculateIdxTy);

	FunctionType *EmbeddingLnTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("EmbeddingLn", EmbeddingLnTy);

	FunctionType *EmbeddingLn_CreateTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("EmbeddingLn_Create", EmbeddingLn_CreateTy);

	FunctionType *cross_entropyTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("cross_entropy", cross_entropyTy);

	FunctionType *cross_entropy_idxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("cross_entropy_idx", cross_entropy_idxTy);

	FunctionType *list_NewTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		true //vararg
	);
	TheModule->getOrInsertFunction("list_New", list_NewTy);

	FunctionType *list_StoreTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("list_Store", list_StoreTy);

	FunctionType *list_printTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("list_print", list_printTy);

	FunctionType *list_LoadTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("list_Load", list_LoadTy);

	FunctionType *list_CreateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("list_Create", list_CreateTy);

	FunctionType *list_IdxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("list_Idx", list_IdxTy);

	FunctionType *assign_wise_list_IdxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("assign_wise_list_Idx", assign_wise_list_IdxTy);

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

	FunctionType *PrintStrVecTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("PrintStrVec", PrintStrVecTy);

	FunctionType *LenStrVecTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
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
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("str_vec_Idx", str_vec_IdxTy);

	FunctionType *str_vec_CalculateIdxTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("str_vec_CalculateIdx", str_vec_CalculateIdxTy);

	FunctionType *str_vec_printTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_vec_print", str_vec_printTy);

	FunctionType *Conv2dTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("Conv2d", Conv2dTy);

	FunctionType *Conv2d_CreateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("Conv2d_Create", Conv2d_CreateTy);

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

	FunctionType *mseTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("mse", mseTy);

	FunctionType *mse_with_prioritiesTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("mse_with_priorities", mse_with_prioritiesTy);

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

	FunctionType *float_vec_Store_IdxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_vec_Store_Idx", float_vec_Store_IdxTy);

	FunctionType *arange_floatTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("arange_float", arange_floatTy);

	FunctionType *zeros_vecTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("zeros_vec", zeros_vecTy);

	FunctionType *ones_vecTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("ones_vec", ones_vecTy);

	FunctionType *float_vec_IdxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("float_vec_Idx", float_vec_IdxTy);

	FunctionType *float_vec_Idx_numTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("float_vec_Idx_num", float_vec_Idx_numTy);

	FunctionType *float_vec_CalculateIdxTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
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

	FunctionType *float_vec_Split_ParallelTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_vec_Split_Parallel", float_vec_Split_ParallelTy);

	FunctionType *float_vec_Split_Strided_ParallelTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_vec_Split_Strided_Parallel", float_vec_Split_Strided_ParallelTy);

	FunctionType *float_vec_sizeTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_vec_size", float_vec_sizeTy);

	FunctionType *print_randomsTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("print_randoms", print_randomsTy);

	FunctionType *randintTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("randint", randintTy);

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

	FunctionType *tensor_opaTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tensor_opa", tensor_opaTy);

	FunctionType *gpuTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("gpu", gpuTy);

	FunctionType *tensor_gpuwTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
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
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("write_zerosw", write_zeroswTy);

	FunctionType *tensor_viewTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("tensor_view", tensor_viewTy);

	FunctionType *tensor_CalculateIdxTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("tensor_CalculateIdx", tensor_CalculateIdxTy);

	FunctionType *zeros_likeTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("zeros_like", zeros_likeTy);

	FunctionType *tensor_printTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tensor_print", tensor_printTy);

	FunctionType *clean_forwardTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("clean_forward", clean_forwardTy);

	FunctionType *save_as_binTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("save_as_bin", save_as_binTy);

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

	FunctionType *Add_To_NotesVector_floatTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("Add_To_NotesVector_float", Add_To_NotesVector_floatTy);

	FunctionType *Add_To_NotesVector_intTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("Add_To_NotesVector_int", Add_To_NotesVector_intTy);

	FunctionType *Add_To_NotesVector_strTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("Add_To_NotesVector_str", Add_To_NotesVector_strTy);

	FunctionType *repeat_interleaveTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("repeat_interleave", repeat_interleaveTy);

	FunctionType *mean_tensorTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("mean_tensor", mean_tensorTy);

	FunctionType *sumTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("sum", sumTy);

	FunctionType *prodTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("prod", prodTy);

	FunctionType *gatherTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("gather", gatherTy);

	FunctionType *tensor_onehotTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("tensor_onehot", tensor_onehotTy);

	FunctionType *priority_sampleTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("priority_sample", priority_sampleTy);

	FunctionType *priority_sample_valTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("priority_sample_val", priority_sample_valTy);

	FunctionType *importance_sample_idxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("importance_sample_idx", importance_sample_idxTy);

	FunctionType *importance_sample_weightTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("importance_sample_weight", importance_sample_weightTy);

	FunctionType *tmaxTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("tmax", tmaxTy);

	FunctionType *tensor_argmaxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("tensor_argmax", tensor_argmaxTy);

	FunctionType *topkTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("topk", topkTy);

	FunctionType *CopyArgTensorTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("CopyArgTensor", CopyArgTensorTy);

	FunctionType *tensor_float_multTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("tensor_float_mult", tensor_float_multTy);

	FunctionType *tensor_float_divTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("tensor_float_div", tensor_float_divTy);

	FunctionType *tensor_float_addTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("tensor_float_add", tensor_float_addTy);

	FunctionType *tensor_float_subTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("tensor_float_sub", tensor_float_subTy);

	FunctionType *tensor_float_equalTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("tensor_float_equal", tensor_float_equalTy);

	FunctionType *tensor_float_diffTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("tensor_float_diff", tensor_float_diffTy);

	FunctionType *tensor_float_minorTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("tensor_float_minor", tensor_float_minorTy);

	FunctionType *tensor_float_minor_eqTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("tensor_float_minor_eq", tensor_float_minor_eqTy);

	FunctionType *tensor_float_higherTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("tensor_float_higher", tensor_float_higherTy);

	FunctionType *tensor_float_higher_eqTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("tensor_float_higher_eq", tensor_float_higher_eqTy);

	FunctionType *opa_gangnam_styleTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("opa_gangnam_style", opa_gangnam_styleTy);

	FunctionType *LinearTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("Linear", LinearTy);

	FunctionType *Linear_LoadTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("Linear_Load", Linear_LoadTy);

	FunctionType *Linear_weightTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("Linear_weight", Linear_weightTy);

	FunctionType *Linear_CreateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("Linear_Create", Linear_CreateTy);

	FunctionType *CosineLRTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("CosineLR", CosineLRTy);

	FunctionType *dir_existsTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("dir_exists", dir_existsTy);

	FunctionType *path_existsTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("path_exists", path_existsTy);

	FunctionType *AdamWTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("AdamW", AdamWTy);

	FunctionType *int_vec_CreateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_vec_Create", int_vec_CreateTy);

	FunctionType *int_vec_LoadTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_vec_Load", int_vec_LoadTy);

	FunctionType *int_vec_StoreTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_vec_Store", int_vec_StoreTy);

	FunctionType *int_vec_Store_IdxTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_vec_Store_Idx", int_vec_Store_IdxTy);

	FunctionType *arange_intTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("arange_int", arange_intTy);

	FunctionType *zeros_intTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("zeros_int", zeros_intTy);

	FunctionType *ones_intTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("ones_int", ones_intTy);

	FunctionType *int_vec_IdxTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("int_vec_Idx", int_vec_IdxTy);

	FunctionType *int_vec_Idx_numTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("int_vec_Idx_num", int_vec_Idx_numTy);

	FunctionType *int_vec_CalculateIdxTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("int_vec_CalculateIdx", int_vec_CalculateIdxTy);

	FunctionType *int_vec_first_nonzeroTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_vec_first_nonzero", int_vec_first_nonzeroTy);

	FunctionType *int_vec_printTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_vec_print", int_vec_printTy);

	FunctionType *int_vec_Split_ParallelTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_vec_Split_Parallel", int_vec_Split_ParallelTy);

	FunctionType *int_vec_Split_Strided_ParallelTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_vec_Split_Strided_Parallel", int_vec_Split_Strided_ParallelTy);

	FunctionType *int_vec_sizeTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_vec_size", int_vec_sizeTy);

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

	FunctionType *str_int_addTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("str_int_add", str_int_addTy);

	FunctionType *str_float_addTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("str_float_add", str_float_addTy);

	FunctionType *int_str_addTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_str_add", int_str_addTy);

	FunctionType *float_str_addTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_str_add", float_str_addTy);

	FunctionType *PrintStrTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("PrintStr", PrintStrTy);

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
		{int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
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

	FunctionType *print_codegenTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("print_codegen", print_codegenTy);

	FunctionType *save_as_intTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("save_as_int", save_as_intTy);

	FunctionType *OneCycleLRTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("OneCycleLR", OneCycleLRTy);

	FunctionType *save_imgTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("save_img", save_imgTy);

	FunctionType *LockMutexTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("LockMutex", LockMutexTy);

	FunctionType *UnlockMutexTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("UnlockMutex", UnlockMutexTy);

	FunctionType *dive_voidTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("dive_void", dive_voidTy);

	FunctionType *emerge_voidTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("emerge_void", emerge_voidTy);

	FunctionType *_tidTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("_tid", _tidTy);

	FunctionType *tidTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tid", tidTy);

	FunctionType *pthread_create_auxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("pthread_create_aux", pthread_create_auxTy);

	FunctionType *pthread_join_auxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("pthread_join_aux", pthread_join_auxTy);

	FunctionType *minTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("min", minTy);

	FunctionType *maxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("max", maxTy);

	FunctionType *logE2fTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("logE2f", logE2fTy);

	FunctionType *roundETy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("roundE", roundETy);

	FunctionType *floorETy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("floorE", floorETy);

	FunctionType *logical_notTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("logical_not", logical_notTy);

	FunctionType *build_vocabTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("build_vocab", build_vocabTy);

	FunctionType *wtokenize_pad_left_idxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("wtokenize_pad_left_idx", wtokenize_pad_left_idxTy);

	FunctionType *__slee_p_Ty= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("__slee_p_", __slee_p_Ty);

	FunctionType *random_sleepTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("random_sleep", random_sleepTy);

	FunctionType *silent_sleepTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("silent_sleep", silent_sleepTy);

	FunctionType *start_timerTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("start_timer", start_timerTy);

	FunctionType *end_timerTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("end_timer", end_timerTy);

	FunctionType *EmbeddingTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("Embedding", EmbeddingTy);

	FunctionType *Embedding_CreateTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("Embedding_Create", Embedding_CreateTy);

	FunctionType *SGDTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("SGD", SGDTy);

	FunctionType *BatchNorm2dTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("BatchNorm2d", BatchNorm2dTy);

	FunctionType *BatchNorm2d_CreateTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("BatchNorm2d_Create", BatchNorm2d_CreateTy);

	FunctionType *load_binTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("load_bin", load_binTy);

	FunctionType *load_bin_idxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("load_bin_idx", load_bin_idxTy);

	FunctionType *wload_binTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("wload_bin", wload_binTy);

	FunctionType *Pool2dTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("Pool2d", Pool2dTy);

	FunctionType *Pool2d_CreateTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("Pool2d_Create", Pool2d_CreateTy);

	FunctionType *MHSAForwardTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("MHSAForward", MHSAForwardTy);

	FunctionType *CreateMHSAOnDemandTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("CreateMHSAOnDemand", CreateMHSAOnDemandTy);

	FunctionType *FirstArgOnDemandTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("FirstArgOnDemand", FirstArgOnDemandTy);

}