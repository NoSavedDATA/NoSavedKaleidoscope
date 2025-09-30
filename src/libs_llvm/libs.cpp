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

	FunctionType *offset_object_ptrTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("offset_object_ptr", offset_object_ptrTy);

	FunctionType *object_Attr_floatTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("object_Attr_float", object_Attr_floatTy);

	FunctionType *object_Attr_intTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("object_Attr_int", object_Attr_intTy);

	FunctionType *object_Load_floatTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("object_Load_float", object_Load_floatTy);

	FunctionType *object_Load_intTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("object_Load_int", object_Load_intTy);

	FunctionType *object_Load_slotTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("object_Load_slot", object_Load_slotTy);

	FunctionType *tie_object_to_objectTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tie_object_to_object", tie_object_to_objectTy);

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

	FunctionType *read_intTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("read_int", read_intTy);

	FunctionType *int_to_strTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("int_to_str", int_to_strTy);

	FunctionType *channel_CreateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("channel_Create", channel_CreateTy);

	FunctionType *str_channel_messageTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_channel_message", str_channel_messageTy);

	FunctionType *channel_str_messageTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("channel_str_message", channel_str_messageTy);

	FunctionType *str_channel_IdxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("str_channel_Idx", str_channel_IdxTy);

	FunctionType *str_channel_terminateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_channel_terminate", str_channel_terminateTy);

	FunctionType *str_channel_aliveTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_channel_alive", str_channel_aliveTy);

	FunctionType *float_channel_messageTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_channel_message", float_channel_messageTy);

	FunctionType *channel_float_messageTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("channel_float_message", channel_float_messageTy);

	FunctionType *float_channel_IdxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("float_channel_Idx", float_channel_IdxTy);

	FunctionType *float_channel_sumTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_channel_sum", float_channel_sumTy);

	FunctionType *float_channel_meanTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_channel_mean", float_channel_meanTy);

	FunctionType *float_channel_terminateTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_channel_terminate", float_channel_terminateTy);

	FunctionType *float_channel_aliveTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_channel_alive", float_channel_aliveTy);

	FunctionType *int_channel_messageTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_channel_message", int_channel_messageTy);

	FunctionType *channel_int_messageTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("channel_int_message", channel_int_messageTy);

	FunctionType *int_channel_IdxTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("int_channel_Idx", int_channel_IdxTy);

	FunctionType *int_channel_sumTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_channel_sum", int_channel_sumTy);

	FunctionType *int_channel_meanTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_channel_mean", int_channel_meanTy);

	FunctionType *int_channel_terminateTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_channel_terminate", int_channel_terminateTy);

	FunctionType *int_channel_aliveTy= FunctionType::get(
		Type::getInt1Ty(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_channel_alive", int_channel_aliveTy);

	FunctionType *Delete_PtrTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("Delete_Ptr", Delete_PtrTy);

	FunctionType *RandomStrOnDemandTy= FunctionType::get(
		int8PtrTy,
		{},
		false
	);
	TheModule->getOrInsertFunction("RandomStrOnDemand", RandomStrOnDemandTy);

	FunctionType *GetEmptyCharTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
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
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("CopyString", CopyStringTy);

	FunctionType *ConcatStrTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("ConcatStr", ConcatStrTy);

	FunctionType *ConcatStrFreeLeftTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("ConcatStrFreeLeft", ConcatStrFreeLeftTy);

	FunctionType *ConcatFloatToStrTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("ConcatFloatToStr", ConcatFloatToStrTy);

	FunctionType *ConcatNumToStrFreeTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("ConcatNumToStrFree", ConcatNumToStrFreeTy);

	FunctionType *MarkToSweep_MarkTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("MarkToSweep_Mark", MarkToSweep_MarkTy);

	FunctionType *MarkToSweep_Mark_ScopefulTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("MarkToSweep_Mark_Scopeful", MarkToSweep_Mark_ScopefulTy);

	FunctionType *MarkToSweep_Mark_ScopelessTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("MarkToSweep_Mark_Scopeless", MarkToSweep_Mark_ScopelessTy);

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

	FunctionType *scope_struct_specTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_spec", scope_struct_specTy);

	FunctionType *set_scope_lineTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("set_scope_line", set_scope_lineTy);

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

	FunctionType *scope_struct_Clear_GC_RootTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Clear_GC_Root", scope_struct_Clear_GC_RootTy);

	FunctionType *scope_struct_Add_GC_RootTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Add_GC_Root", scope_struct_Add_GC_RootTy);

	FunctionType *scope_struct_Add_PointerTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("scope_struct_Add_Pointer", scope_struct_Add_PointerTy);

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

	FunctionType *nsk_vec_sizeTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("nsk_vec_size", nsk_vec_sizeTy);

	FunctionType *__idx__Ty= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("__idx__", __idx__Ty);

	FunctionType *__sliced_idx__Ty= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("__sliced_idx__", __sliced_idx__Ty);

	FunctionType *_quit_Ty= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("_quit_", _quit_Ty);

	FunctionType *read_floatTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("read_float", read_floatTy);

	FunctionType *float_to_strTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("float_to_str", float_to_strTy);

	FunctionType *nullptr_getTy= FunctionType::get(
		int8PtrTy,
		{},
		false
	);
	TheModule->getOrInsertFunction("nullptr_get", nullptr_getTy);

	FunctionType *is_nullTy= FunctionType::get(
		Type::getInt1Ty(*TheContext),
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("is_null", is_nullTy);

	FunctionType *printTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("print", printTy);

	FunctionType *dict_CreateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("dict_Create", dict_CreateTy);

	FunctionType *dict_NewTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		true //vararg
	);
	TheModule->getOrInsertFunction("dict_New", dict_NewTy);

	FunctionType *dict_Store_KeyTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("dict_Store_Key", dict_Store_KeyTy);

	FunctionType *dict_Store_Key_intTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("dict_Store_Key_int", dict_Store_Key_intTy);

	FunctionType *dict_Store_Key_floatTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("dict_Store_Key_float", dict_Store_Key_floatTy);

	FunctionType *dict_printTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("dict_print", dict_printTy);

	FunctionType *dict_QueryTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("dict_Query", dict_QueryTy);

	FunctionType *list_NewTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		true //vararg
	);
	TheModule->getOrInsertFunction("list_New", list_NewTy);

	FunctionType *list_append_intTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("list_append_int", list_append_intTy);

	FunctionType *list_append_floatTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("list_append_float", list_append_floatTy);

	FunctionType *list_append_boolTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getInt1Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("list_append_bool", list_append_boolTy);

	FunctionType *list_appendTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("list_append", list_appendTy);

	FunctionType *list_printTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("list_print", list_printTy);

	FunctionType *tuple_printTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("tuple_print", tuple_printTy);

	FunctionType *list_CreateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("list_Create", list_CreateTy);

	FunctionType *list_sizeTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("list_size", list_sizeTy);

	FunctionType *list_CalculateIdxTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("list_CalculateIdx", list_CalculateIdxTy);

	FunctionType *to_intTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("to_int", to_intTy);

	FunctionType *to_floatTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("to_float", to_floatTy);

	FunctionType *list_CalculateSliceIdxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("list_CalculateSliceIdx", list_CalculateSliceIdxTy);

	FunctionType *list_SliceTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("list_Slice", list_SliceTy);

	FunctionType *assign_wise_list_IdxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("assign_wise_list_Idx", assign_wise_list_IdxTy);

	FunctionType *int_list_Store_IdxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_list_Store_Idx", int_list_Store_IdxTy);

	FunctionType *float_list_Store_IdxTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getFloatTy(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_list_Store_Idx", float_list_Store_IdxTy);

	FunctionType *zipTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		true //vararg
	);
	TheModule->getOrInsertFunction("zip", zipTy);

	FunctionType *list_IdxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("list_Idx", list_IdxTy);

	FunctionType *tuple_IdxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("tuple_Idx", tuple_IdxTy);

	FunctionType *str_vec_CreateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_vec_Create", str_vec_CreateTy);

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

	FunctionType *shuffle_strTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("shuffle_str", shuffle_strTy);

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

	FunctionType *float_vec_CreateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("float_vec_Create", float_vec_CreateTy);

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

	FunctionType *zeros_floatTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("zeros_float", zeros_floatTy);

	FunctionType *ones_floatTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("ones_float", ones_floatTy);

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

	FunctionType *allocate_voidTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("allocate_void", allocate_voidTy);

	FunctionType *print_randomsTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("print_randoms", print_randomsTy);

	FunctionType *randintTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("randint", randintTy);

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

	FunctionType *int_vec_CreateTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_vec_Create", int_vec_CreateTy);

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

	FunctionType *rand_int_vecTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("rand_int_vec", rand_int_vecTy);

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

	FunctionType *int_vec_CalculateSliceIdxTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		true //vararg
	);
	TheModule->getOrInsertFunction("int_vec_CalculateSliceIdx", int_vec_CalculateSliceIdxTy);

	FunctionType *int_vec_SliceTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("int_vec_Slice", int_vec_SliceTy);

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

	FunctionType *str_CopyArgTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_CopyArg", str_CopyArgTy);

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

	FunctionType *str_bool_addTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, Type::getInt1Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("str_bool_add", str_bool_addTy);

	FunctionType *bool_str_addTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt1Ty(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("bool_str_add", bool_str_addTy);

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

	FunctionType *str_DeleteTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("str_Delete", str_DeleteTy);

	FunctionType *readlineTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("readline", readlineTy);

	FunctionType *print_codegenTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("print_codegen", print_codegenTy);

	FunctionType *print_codegen_silentTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("print_codegen_silent", print_codegen_silentTy);

	FunctionType *bool_to_strTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt1Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("bool_to_str", bool_to_strTy);

	FunctionType *get_barrierTy= FunctionType::get(
		int8PtrTy,
		{Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("get_barrier", get_barrierTy);

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
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("dive_void", dive_voidTy);

	FunctionType *dive_intTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("dive_int", dive_intTy);

	FunctionType *dive_floatTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("dive_float", dive_floatTy);

	FunctionType *emerge_voidTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("emerge_void", emerge_voidTy);

	FunctionType *emerge_intTy= FunctionType::get(
		Type::getInt32Ty(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("emerge_int", emerge_intTy);

	FunctionType *emerge_floatTy= FunctionType::get(
		Type::getFloatTy(*TheContext),
		{int8PtrTy, int8PtrTy},
		false
	);
	TheModule->getOrInsertFunction("emerge_float", emerge_floatTy);

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

	FunctionType *FirstArgOnDemandTy= FunctionType::get(
		int8PtrTy,
		{int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
		false
	);
	TheModule->getOrInsertFunction("FirstArgOnDemand", FirstArgOnDemandTy);

}