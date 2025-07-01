

#include <map>
#include <string>
#include <vector>
#include "../include.h"
#include "../clean_up/clean_up.h"
#include "../compiler_frontend/include.h"

std::vector<std::string> user_cpp_functions;

void set_user_functions() {

    user_cpp_functions = {"Linear", "Conv2d", "tensor_view", "tensor_clip", "tensor_argmax", "tensor_tmax", "tensor_onehot", "tensor_shape", "tensor_permute", "tensor_cpu", "printtt",
        "tensor_sum", "tensor_prod", "tensor_mean", "mean_tensor", "tensor_tmin", "tensor_argmin", "tensor_topk", "tensor_repeat_interleave",
        "tensor_save_img", "tensor_gpu", "tensor_gpuw", "tensor_save_as_int", "tensor_save_as_bin", "tensor_gather", "str_split_idx", "str_to_float", "list_print",
        "BatchNorm2d", "Pool2d", "LSTM", "MHSA", "Embedding", "list_test",
		"str_Create", "str_Load", "str_Store", "str_MarkToSweep", "str_Copy", "str_str_add", "str_float_add", "float_str_add", "PrintStr", "cat_str_float", "SplitString", "str_split_idx", "str_to_float", "StrToFloat", "str_Delete", 
		"int_Create", "int_Load", "int_Store", 
		"Pool2d", "Pool2d_Create", 
		"pinned_tensor_Create", "pinned_tensor_Load", "pinned_tensor_Store_Idx", "pinned_tensor_CalculateIdx", 
		"save_as_int", 
		"MHSAForward", "CreateMHSAOnDemand", 
		"print", 
		"list_New", "list_Store", "list_print", "list_Load", "list_Create", "list_Idx", "assign_wise_list_Idx", 
		"__slee_p_", "random_sleep", "silent_sleep", "start_timer", "end_timer", 
		"CosineLR", 
		"relu", "gelu", "sigmoid", "_tanh", "softmax", 
		"AttrTensorNoFree", "AttrTensorOnIdx", "AttrTensorOnIdxTensor", "AttrPinnedFromTensorOnIdx", "IdxTensor", "IdxTensorWithTensor", 
		"RandomCrop", "RandomHorizontalFlip", "NormalizeImg", "Jitter", 
		"tensor_tensor_mma", "tensor_tensor_add", "tensor_tensor_sub", "tensor_tensor_equal", "tensor_tensor_mult", "tensor_tensor_div", 
		"str_vec_Create", "str_vec_Load", "str_vec_Store", "PrintStrVec", "LenStrVec", "ShuffleStrVec", "shuffle_str", "_glob_b_", "IndexStrVec", "str_vec_Idx", "str_vec_CalculateIdx", "str_vec_print", 
		"AdamW", 
		"load_bin", "load_bin_idx", "wload_bin", 
		"LSTM", "LSTM_Create", 
		"tensor_onehot", "priority_sample", "priority_sample_val", "importance_sample_idx", "importance_sample_weight", "tmax", "tensor_argmax", "topk", 
		"nullptr_get", "check_is_null", 
		"repeat_interleave", "mean_tensor", "tensor_mean", "sum", "prod", "gather", 
		"dive_void", "emerge_void", "_tid", "tid", "pthread_create_aux", "pthread_join_aux", 
		"Embedding", "Embedding_Create", 
		"BatchNorm2d", "BatchNorm2d_Create", 
		"tensor_float_mult", "tensor_float_div", "tensor_float_add", "tensor_float_sub", "tensor_float_equal", "tensor_float_diff", "tensor_float_minor", "tensor_float_minor_eq", "tensor_float_higher", "tensor_float_higher_eq", "opa_gangnam_style", 
		"print_randoms", "randint", 
		"tensor_Create", "tensor_Load", "tensor_Copy", "tensor_StoreTrigger", "gpu", "tensor_gpuw", "cpu", "cpu_idx", "randu_like", "write_zerosw", "tensor_view", "tensor_CalculateIdx", "zeros_like", "tensor_CopyArg", "tensor_print", 
		"ConcatScopeStr", 
		"load_img", "gload_img", "wload_img", "wload_img_resize", "load_preprocess_img", 
		"LockMutex", "UnlockMutex", 
		"FirstArgOnDemand", 
		"EmbeddingLn", "EmbeddingLn_Create", 
		"InstantiateObject", "objHash", "LoadObject", "InitObjectVecWithNull", "is_null", "objAttr_var_from_var", "objAttr_var_from_vec", "objAttr_vec_from_var", "objAttr_vec_from_vec", "append", "LoadObjectScopeName", "object_Attr_on_Offset_float", "object_Attr_on_Offset_int", "object_Attr_on_Offset", "object_Load_on_Offset_float", "object_Load_on_Offset_int", "object_Load_on_Offset", "object_ptr_Load_on_Offset", "object_ptr_Attribute_object", 
		"mse", "mse_with_priorities", 
		"min", "max", "logE2f", "roundE", "floorE", "logical_not", 
		"str_Create", "str_Load", "str_Store", "str_Copy", "str_str_add", "str_int_add", "str_float_add", "int_str_add", "float_str_add", "PrintStr", "cat_str_float", "SplitString", "str_split_idx", "str_to_float", "StrToFloat", "str_Delete", 
		"backprop", 
		"PrintTensor", "PrintTensorF", "PrintTensorI8", 
		"network_ema", 
		"rl_discounted_return", 
		"build_vocab", "wtokenize_pad_left_idx", 
		"float_vec_Create", "float_vec_Load", "float_vec_Store", "float_vec_Store_Idx", "arange_float", "zeros_vec", "ones_vec", "float_vec_Idx", "float_vec_Idx_num", "float_vec_CalculateIdx", "float_vec_first_nonzero", "float_vec_print", "float_vec_Split_Parallel", "float_vec_Split_Strided_Parallel", "float_vec_size", 
		"CreateNotesVector", "Dispose_NotesVector", "Add_To_NotesVector_float", "Add_To_NotesVector_int", "Add_To_NotesVector_str", 
		"RandomStrOnDemand", "GetEmptyChar", "FreeCharFromFunc", "FreeChar", "CopyString", "ConcatStr", "ConcatStrFreeLeft", "ConcatStrFreeRight", "ConcatStrFree", "ConcatFloatToStr", "ConcatNumToStrFree", 
		"tensor_transpose", 
		"save_img", 
		"OneCycleLR", 
		"MarkToSweep_Mark", "MarkToSweep_Unmark_Scopeful", "MarkToSweep_Unmark_Scopeless", 
		"save_as_bin", 
		"cross_entropy", "cross_entropy_idx", 
		"int_vec_Create", "int_vec_Load", "int_vec_Store", "int_vec_Store_Idx", "arange_int", "zeros_int", "ones_int", "int_vec_Idx", "int_vec_Idx_num", "int_vec_CalculateIdx", "int_vec_first_nonzero", "int_vec_print", "int_vec_Split_Parallel", "int_vec_Split_Strided_Parallel", "int_vec_size", 
		"PrintDims", "StoreDimsOnDemand", "CalculateIdxOffset", "tensor_shape", 
		"scope_struct_Create", "scope_struct_Copy", "scope_struct_Overwrite", "scope_struct_Dive", "set_scope_at_return", "set_scope_not_at_return", "set_scope_first_arg", "set_scope_scope", "set_scope_thread_id", "set_scope_has_grad", "set_scope_function_name", "get_scope_first_arg", "get_scope_scope", "get_scope_thread_id", "get_scope_has_grad", "scope_struct_Reset_Threads", "scope_struct_Increment_Thread", "set_scope_object", "get_scope_object", "scope_struct_Save_for_Async", "scope_struct_Load_for_Async", "scope_struct_Store_Asyncs_Count", "scope_struct_Print", "scope_struct_Get_Async_Scope", "scope_struct_Alloc_MarkSweepMap", "scope_struct_Copy_MarkSweepMap", "scope_struct_Sweep", "scope_struct_Clean_Scope", "scope_struct_Delete", 
		"print_codegen", 
		"logE", "logE2", "clip", 
		"PrintFloat", "UnbugFloat", "print_float", "float_Create", "float_Load", "float_Store", 
		"Linear", "Linear_Load", "Linear_weight", "Linear_Create", 
		"Conv2d", "Conv2d_Create", 
		"dir_exists", "path_exists", 
		"SGD", 
		"clean_forward", 
		"_exit", 
		"dictionary_Create", "dictionary_Dispose", 

	};


	clean_up_functions["tensor"] = tensor_Clean_Up;

	clean_up_functions["list"] = list_Clean_Up;

	clean_up_functions["int_vec"] = int_vec_Clean_Up;

	clean_up_functions["float_vec"] = float_vec_Clean_Up;

	clean_up_functions["str"] = str_Clean_Up;

	clean_up_functions["int"] = int_Clean_Up;

	clean_up_functions["float"] = float_Clean_Up;

	clean_up_functions["str_vec"] = str_vec_Clean_Up;

	backward_functions["linear_backward"] = linear_backward;

	backward_functions["scalarmult_backward"] = scalarmult_backward;

	backward_functions["mhsa_backward"] = mhsa_backward;

	backward_functions["relu_backward"] = relu_backward;

	backward_functions["gelu_backward"] = gelu_backward;

	backward_functions["sigmoid_backward"] = sigmoid_backward;

	backward_functions["tanh_backward"] = tanh_backward;

	backward_functions["embedding_backward"] = embedding_backward;

	backward_functions["batchnorm2d_backward"] = batchnorm2d_backward;

	backward_functions["lstm_backward"] = lstm_backward;

	backward_functions["pool2d_backward"] = pool2d_backward;

	backward_functions["mean_over_semilast_dim_backward"] = mean_over_semilast_dim_backward;

	backward_functions["gather_last_dim_backward"] = gather_last_dim_backward;

	backward_functions["conv2d_backward"] = conv2d_backward;

	backward_functions["embeddingln_backward"] = embeddingln_backward;


}