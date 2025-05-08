

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
		"Pool2d", "Pool2d_Create", 
		"pinned_tensor_Create", "pinned_tensor_Store_Idx", "pinned_tensor_CalculateIdx", 
		"save_as_int", 
		"MHSAForward", "CreateMHSAOnDemand", 
		"print", 
		"list_New", "list_Store", "list_Load", "list_print", "list_checkmate", "list_test", "list_Create", "list_MarkToSweep", "list_Idx", "assign_wise_list_Idx", 
		"__slee_p_", "silent_sleep", "start_timer", "end_timer", 
		"CosineLR", 
		"relu", "gelu", "sigmoid", "_tanh", "softmax", 
		"AttrTensorOnIdx", "AttrTensorOnIdxTensor", "AttrPinnedFromTensorOnIdx", "IdxTensor", "IdxTensorWithTensor", 
		"RandomCrop", "RandomHorizontalFlip", "NormalizeImg", "Jitter", 
		"tensor_tensor_mma", "tensor_tensor_add", "tensor_tensor_sub", "tensor_tensor_equal", "tensor_tensor_mult", "tensor_tensor_div", 
		"str_vec_Create", "str_vec_Load", "str_vec_Store", "str_vec_MarkToSweep", "PrintStrVec", "LenStrVec", "ShuffleStrVec", "_glob_b_", "IndexStrVec", "str_vec_Idx", "str_vec_CalculateIdx", "str_vec_print", 
		"AdamW", 
		"load_bin", "load_bin_idx", "wload_bin", 
		"LSTMForward", "CreateLSTMOnDemand", 
		"tensor_onehot", "priority_sample", "priority_sample_val", "importance_sample_idx", "importance_sample_weight", "tmax", "tensor_argmax", "topk", 
		"nullptr_get", 
		"repeat_interleave", "mean_tensor", "sum", "prod", "gather", 
		"EmbeddingForward", "CreateEmbeddingOnDemand", 
		"BatchNorm2d", "BatchNorm2d_Create", 
		"tensor_float_mult", "tensor_float_div", "tensor_float_add", "tensor_float_sub", "tensor_float_equal", "tensor_float_diff", "tensor_float_minor", "tensor_float_minor_eq", "tensor_float_higher", "tensor_float_higher_eq", "opa_gangnam_style", 
		"print_randoms", "randint", 
		"tensor_Create", "tensor_Load", "tensor_Store", "tensor_MarkToSweep", "gpu", "tensor_gpuw", "cpu", "cpu_idx", "randu_like", "write_zerosw", "tensor_view", "NewVecToTensor", "tensor_CalculateIdx", "zeros_like", 
		"ConcatScopeStr", "ConcatScopeAtCallExpr", "AddFloatToScopeCleanList", "AddToScopeCleanList", "CleanScopeVars", "RemoveTensorScope", "RemoveTensorScopeAttrOnIndex", 
		"load_img", "gload_img", "wload_img", "wload_img_resize", "load_preprocess_img", 
		"LockMutex", "UnlockMutex", 
		"FirstArgOnDemand", 
		"InstantiateObject", "objHash", "LoadObject", "InitObjectVecWithNull", "is_null", "objAttr_var_from_var", "objAttr_var_from_vec", "objAttr_vec_from_var", "objAttr_vec_from_vec", "append", "LoadObjectScopeName", 
		"mse", "mse_with_priorities", 
		"min", "max", "logE2f", "roundE", "floorE", "logical_not", 
		"str_Create", "str_Load", "str_Store", "str_MarkToSweep", "str_Copy", "str_str_add", "str_float_add", "float_str_add", "PrintStr", "cat_str_float", "SplitString", "str_split_idx", "str_to_float", "StrToFloat", "str_Delete", 
		"CopyArgTensor", 
		"backprop", 
		"PrintTensor", "print_tensor", "PrintTensorF", 
		"network_ema", 
		"rl_discounted_return", 
		"build_vocab", "tokenize", "wtokenize", "wtokenize_pad_left", "wtokenize_pad_left_batch_first", "wtokenize_pad_left_idx", 
		"float_vec_Create", "float_vec_Load", "float_vec_Store", "float_vec_MarkToSweep", "float_vec_Store_Idx", "PrintFloatVec", "zeros_vec", "ones_vec", "float_vec_Idx", "float_vec_CalculateIdx", "float_vec_first_nonzero", "float_vec_print", 
		"CreateNotesVector", "Dispose_NotesVector", "Add_Float_To_NotesVector", "Add_String_To_NotesVector", 
		"pthread_create_aux", "pthread_join_aux", 
		"RandomStrOnDemand", "GetEmptyChar", "FreeCharFromFunc", "FreeChar", "CopyString", "ConcatStr", "ConcatStrFreeLeft", "ConcatStrFreeRight", "ConcatStrFree", "ConcatFloatToStr", "ConcatNumToStrFree", 
		"save_img", 
		"OneCycleLR", 
		"save_as_bin", 
		"cross_entropy", "cross_entropy_idx", 
		"btc_mult", "btc_multT", 
		"PrintDims", "NewDimsOnIdx", "StoreDimsOnDemand", "CalculateIdxOffset", "tensor_shape", 
		"scope_struct_Create", "scope_struct_Copy", "scope_struct_Overwrite", "scope_struct_Dive", "set_scope_first_arg", "set_scope_scope", "set_scope_previous_scope", "set_scope_thread_id", "set_scope_has_grad", "set_scope_function_name", "get_scope_first_arg", "get_scope_scope", "get_scope_previous_scope", "get_scope_thread_id", "get_scope_has_grad", "scope_struct_Save_for_Async", "scope_struct_Load_for_Async", "scope_struct_Print", "scope_struct_Get_Async_Scope", "scope_struct_Alloc_MarkSweepMap", "scope_struct_Copy_MarkSweepMap", "scope_struct_Clean_Scope", "scope_struct_Delete", 
		"print_codegen", 
		"logE", "logE2", "clip", 
		"PrintFloat", "UnbugFloat", "print_float", "float_Create", "float_Load", "float_Store", "float_MarkToSweep", "StoreOnDemandNoFree", "LoadOnDemandNoFree", 
		"Linear", "Linear_Create", 
		"Conv2d", "Conv2d_Create", 
		"dir_exists", "path_exists", 
		"SGD", 
		"clean_forward", 
		"_exit", 
		"dictionary_Create", "dictionary_Dispose", 

	};


	clean_up_functions["tensor"] = tensor_Clean_Up;

	clean_up_functions["list"] = list_Clean_Up;

	clean_up_functions["float_vec"] = float_vec_Clean_Up;

	clean_up_functions["str"] = str_Clean_Up;

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


}