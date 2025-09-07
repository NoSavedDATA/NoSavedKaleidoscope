

#include <map>
#include <string>
#include <vector>

#include "../include.h"
#include "../clean_up/clean_up.h"
#include "../compiler_frontend/include.h"

std::vector<std::string> user_cpp_functions;

void set_user_functions() {

    user_cpp_functions = {"Linear", "tensor_view", "tensor_clip", "tensor_argmax", "tensor_tmax", "tensor_onehot", "tensor_shape", "tensor_permute", "tensor_cpu", "printtt",
        "tensor_sum", "tensor_prod", "tensor_mean", "mean_tensor", "tensor_tmin", "tensor_argmin", "tensor_topk", "tensor_repeat_interleave",
        "tensor_save_img", "tensor_gpu", "tensor_gpuw", "tensor_save_as_int", "tensor_save_as_bin", "tensor_gather", "str_split_idx", "str_to_float", "list_print",
		"int_Create", "int_Load", "int_Store", 
		"get_barrier", 
		"print", 
		"list_New", "list_append_int", "list_append_float", "list_append_bool", "list_append", "list_print", "list_Create", "list_size", "list_CalculateIdx", "list_Idx", "to_int", "to_float", "list_CalculateSliceIdx", "list_Slice", "assign_wise_list_Idx", "int_list_Store_Idx", "float_list_Store_Idx", 
		"__slee_p_", "random_sleep", "silent_sleep", "start_timer", "end_timer", 
		"str_vec_Create", "LenStrVec", "ShuffleStrVec", "shuffle_str", "IndexStrVec", "str_vec_Idx", "str_vec_CalculateIdx", "str_vec_print", 
		"nullptr_get", "check_is_null", 
		"dive_void", "dive_int", "dive_float", "emerge_void", "emerge_int", "emerge_float", "_tid", "tid", "pthread_create_aux", "pthread_join_aux", "pthread_create_aux", "pthread_join_aux", 
		"Delete_Ptr", 
		"print_randoms", "randint", 
		"LockMutex", "UnlockMutex", 
		"FirstArgOnDemand", 
		"InstantiateObject", "objHash", "LoadObject", "InitObjectVecWithNull", "is_null", "objAttr_var_from_var", "objAttr_var_from_vec", "objAttr_vec_from_var", "objAttr_vec_from_vec", "append", "LoadObjectScopeName", "offset_object_ptr", "object_Attr_float", "object_Attr_int", "object_Load_float", "object_Load_int", "object_Load_slot", "tie_object_to_object", "object_Attr_on_Offset_float", "object_Attr_on_Offset_int", "object_Attr_on_Offset", "object_Load_on_Offset_float", "object_Load_on_Offset_int", "object_Load_on_Offset", "object_ptr_Load_on_Offset", "object_ptr_Attribute_object", 
		"min", "max", "logE2f", "roundE", "floorE", "logical_not", 
		"str_Create", "str_Store", "str_Copy", "str_CopyArg", "str_str_add", "str_int_add", "str_float_add", "int_str_add", "float_str_add", "str_bool_add", "bool_str_add", "PrintStr", "cat_str_float", "str_split_idx", "str_to_float", "str_Delete", 
		"channel_Create", "str_channel_message", "channel_str_message", "str_channel_Idx", "str_channel_terminate", "str_channel_alive", "float_channel_message", "channel_float_message", "float_channel_Idx", "float_channel_sum", "float_channel_mean", "float_channel_terminate", "float_channel_alive", "int_channel_message", "channel_int_message", "int_channel_Idx", "int_channel_sum", "int_channel_mean", "int_channel_terminate", "int_channel_alive", 
		"nsk_vec_size", "__idx__", "__sliced_idx__", 
		"float_vec_Create", "float_vec_Store_Idx", "arange_float", "zeros_float", "ones_float", "float_vec_Idx", "float_vec_Idx_num", "float_vec_CalculateIdx", "float_vec_first_nonzero", "float_vec_print", "float_vec_Split_Parallel", "float_vec_Split_Strided_Parallel", "float_vec_size", 
		"CreateNotesVector", "Dispose_NotesVector", "Add_To_NotesVector_float", "Add_To_NotesVector_int", "Add_To_NotesVector_str", 
		"RandomStrOnDemand", "GetEmptyChar", "FreeCharFromFunc", "FreeChar", "CopyString", "ConcatStr", "ConcatStrFreeLeft", "ConcatStrFreeRight", "ConcatStrFree", "ConcatFloatToStr", "ConcatNumToStrFree", 
		"MarkToSweep_Mark", "MarkToSweep_Mark_Scopeful", "MarkToSweep_Mark_Scopeless", "MarkToSweep_Unmark_Scopeful", "MarkToSweep_Unmark_Scopeless", 
		"int_vec_Create", "int_vec_Load", "int_vec_Store", "int_vec_Store_Idx", "earth_cable", "arange_int", "zeros_int", "ones_int", "int_vec_Idx", "int_vec_Idx_num", "int_vec_CalculateIdx", "int_vec_CalculateSliceIdx", "int_vec_Slice", "int_vec_first_nonzero", "int_vec_print", "int_vec_Split_Parallel", "int_vec_Split_Strided_Parallel", "int_vec_size", 
		"set_scope_line", "scope_struct_Create", "scope_struct_Copy", "scope_struct_Overwrite", "scope_struct_Dive", "set_scope_first_arg", "set_scope_scope", "set_scope_thread_id", "set_scope_has_grad", "set_scope_function_name", "get_scope_first_arg", "get_scope_scope", "get_scope_thread_id", "get_scope_has_grad", "scope_struct_Reset_Threads", "scope_struct_Increment_Thread", "set_scope_object", "get_scope_object", "scope_struct_Save_for_Async", "scope_struct_Load_for_Async", "scope_struct_Store_Asyncs_Count", "scope_struct_Print", "scope_struct_Get_Async_Scope", "scope_struct_Alloc_MarkSweepMap", "scope_struct_Copy_MarkSweepMap", "scope_struct_Sweep", "scope_struct_Clean_Scope", "scope_struct_Delete", 
		"print_codegen", "print_codegen_silent", 
		"print_float", 
		"dict_Create", "dict_New", "dict_Store_Key", "dict_Store_Key_int", "dict_Store_Key_float", "dict_print", "dict_Query", 
		"dir_exists", "path_exists", 
		"_quit_", 

	};


	clean_up_functions["dict"] = dict_Clean_Up;

	clean_up_functions["list"] = list_Clean_Up;

	clean_up_functions["int_vec"] = int_vec_Clean_Up;

	clean_up_functions["float_vec"] = float_vec_Clean_Up;

	clean_up_functions["str"] = str_Clean_Up;

	clean_up_functions["str_vec"] = str_vec_Clean_Up;


}