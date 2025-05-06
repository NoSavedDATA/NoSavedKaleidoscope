

#include <map>
#include <string>
#include <vector>

#include "../clean_up/clean_up.h"
#include "../compiler_frontend/include.h"

std::vector<std::string> user_cpp_functions;

void set_user_functions() {

    user_cpp_functions = {"Linear", "Conv2d", "tensor_view", "tensor_clip", "tensor_argmax", "tensor_tmax", "tensor_onehot", "tensor_shape", "tensor_permute", "tensor_cpu", "printtt",
        "tensor_sum", "tensor_prod", "tensor_mean", "mean_tensor", "tensor_tmin", "tensor_argmin", "tensor_topk", "tensor_repeat_interleave",
        "tensor_save_img", "tensor_gpu", "tensor_gpuw", "tensor_save_as_int", "tensor_save_as_bin", "tensor_gather", "str_split_idx", "str_to_float", "list_print",
        "BatchNorm2d", "Pool2d", "LSTM", "MHSA", "Embedding", "list_test",
		"str_Create", "str_Load", "str_Store", "str_MarkToSweep", "str_Copy", "str_str_add", "str_float_add", "float_str_add", "PrintStr", "cat_str_float", "SplitString", "str_split_idx", "str_to_float", "StrToFloat", "str_Delete", 
		"pinned_tensor_Create", "pinned_tensor_Store_Idx", "pinned_tensor_CalculateIdx", 
		"list_New", "list_Store", "list_Load", "list_print", "list_checkmate", "list_test", "list_Create", "list_MarkToSweep", "list_Idx", "assign_wise_list_Idx", 
		"str_vec_Create", "str_vec_Load", "str_vec_Store", "str_vec_MarkToSweep", "PrintStrVec", "LenStrVec", "ShuffleStrVec", "_glob_b_", "IndexStrVec", "str_vec_Idx", "str_vec_CalculateIdx", "str_vec_print", 
		"nullptr_get", 
		"tensor_Create", "tensor_Load", "tensor_Store", "tensor_MarkToSweep", "gpu", "tensor_gpuw", "cpu", "cpu_idx", "randu_like", "write_zerosw", "tensor_view", "NewVecToTensor", "tensor_CalculateIdx", "zeros_like", 
		"InstantiateObject", "objHash", "LoadObject", "InitObjectVecWithNull", "is_null", "objAttr_var_from_var", "objAttr_var_from_vec", "objAttr_vec_from_var", "objAttr_vec_from_vec", "append", "LoadObjectScopeName", 
		"str_Create", "str_Load", "str_Store", "str_MarkToSweep", "str_Copy", "str_str_add", "str_float_add", "float_str_add", "PrintStr", "cat_str_float", "SplitString", "str_split_idx", "str_to_float", "StrToFloat", "str_Delete", 
		"float_vec_Create", "float_vec_Load", "float_vec_Store", "float_vec_MarkToSweep", "float_vec_Store_Idx", "PrintFloatVec", "zeros_vec", "ones_vec", "float_vec_Idx", "float_vec_CalculateIdx", "float_vec_first_nonzero", "float_vec_print", 
		"CreateNotesVector", "Dispose_NotesVector", "Add_Float_To_NotesVector", "Add_String_To_NotesVector", 
		"PrintFloat", "UnbugFloat", "print_float", "float_Create", "float_Load", "float_Store", "float_MarkToSweep", "StoreOnDemandNoFree", "LoadOnDemandNoFree", 
		"dictionary_Create", "dictionary_Dispose", 

	};


	clean_up_functions["tensor"] = tensor_Clean_Up;

	clean_up_functions["list"] = list_Clean_Up;

	clean_up_functions["float_vec"] = float_vec_Clean_Up;

	clean_up_functions["str"] = str_Clean_Up;

	clean_up_functions["float"] = float_Clean_Up;

	clean_up_functions["str_vec"] = str_vec_Clean_Up;


}