

#include <map>
#include <string>
#include <vector>

#include "../compiler_frontend/include.h"


std::map<std::string, std::string> functions_return_type;

void set_functions_return_type() {

  functions_return_type = {{"gelu", "tensor"}, {"sigmoid", "tensor"}, {"_tanh", "tensor"}, {"relu", "tensor"}, {"softmax", "tensor"},
                           {"log", "tensor"}, {"randu_like", "tensor"}, {"RandomCrop", "tensor"}, {"RandomHorizontalFlip", "tensor"}, {"NormalizeImg", "tensor"},
                           {"dropout", "tensor"}, {"rl_discounted_return", "tensor"}, {"self_attn", "tensor"}, {"Jitter", "tensor"}, {"mse_with_priorities", "tensor"},
                           {"btc_mult", "tensor"}, {"btc_multT", "tensor"}, {"tensor_view", "tensor"}, {"clip", "tensor"}, {"tensor_argmax", "tensor"}, {"tmax", "tensor"},
                           {"tensor_onehot", "tensor"}, {"permute", "tensor"}, {"cpu", "tensor"}, {"printtt", "tensor"}, {"sum", "tensor"},
                           {"prod", "tensor"}, {"tensor_mean", "tensor"}, {"tmin", "tensor"}, {"argmin", "tensor"}, {"topk", "tensor"}, {"repeat_interleave", "tensor"},
                           {"save_img", "tensor"}, {"tensor_gpu", "tensor"}, {"save_as_int", "tensor"}, {"save_as_bin", "tensor"}, {"gather", "tensor"},
                           {"to_string", "str"}, {"cat_str_float", "str"}, {"Linear", "tensor"}, {"str_split_idx", "str"}, {"str_to_float", "float"},
                           {"mean_tensor", "tensor"},
                           {"BatchNorm2d", "tensor"}, {"Pool2d", "tensor"}, {"LSTM", "tensor"}, {"MHSA", "tensor"}, {"Embedding", "tensor"},
						{"IndexStrVec", "str"}, {"str_vec_Idx", "str"}, {"ShuffleStrVec", "str_vec"}, {"_glob_b_", "str_vec"},
						{"nsk_vec_size", "int"}, {"__idx__", "int"}, {"__sliced_idx__", "int"}, 
						{"emerge_int", "int"}, {"tid", "int"}, 
						{"str_Create", "str"}, {"str_Load", "str"}, {"str_Copy", "str"}, {"str_CopyArg", "str"}, {"str_str_add", "str"}, {"str_int_add", "str"}, {"str_float_add", "str"}, {"int_str_add", "str"}, {"float_str_add", "str"}, {"cat_str_float", "str"}, {"SplitString", "str_vec"}, {"str_split_idx", "str"}, 
						{"RandomStrOnDemand", "str"}, {"GetEmptyChar", "str"}, {"CopyString", "str"}, {"ConcatStr", "str"}, {"ConcatStrFreeLeft", "str"}, {"ConcatStrFreeRight", "str"}, {"ConcatStrFree", "str"}, {"ConcatFloatToStr", "str"}, {"ConcatNumToStrFree", "str"}, 
						{"float_vec_Load", "float_vec"}, {"arange_float", "float_vec"}, {"zeros_vec", "float_vec"}, {"ones_vec", "float_vec"}, {"float_vec_CalculateIdx", "int"}, {"float_vec_Split_Parallel", "float_vec"}, {"float_vec_Split_Strided_Parallel", "float_vec"}, 
						{"int_vec_Load", "int_vec"}, {"int_vec_Store", "int"}, {"int_vec_Store_Idx", "int"}, {"arange_int", "int_vec"}, {"zeros_int", "int_vec"}, {"ones_int", "int_vec"}, {"int_vec_Idx", "int"}, {"int_vec_Idx_num", "int"}, {"int_vec_CalculateIdx", "int"}, {"int_vec_CalculateSliceIdx", ""}, {"int_vec_Slice", "int_vec"}, {"int_vec_first_nonzero", "int"}, {"int_vec_print", "int"}, {"int_vec_Split_Parallel", "int_vec"}, {"int_vec_Split_Strided_Parallel", "int_vec"}, {"int_vec_size", "int"}, 
						{"scope_struct_Create", ""}, {"scope_struct_Copy", ""}, {"scope_struct_Overwrite", ""}, {"scope_struct_Dive", ""}, {"get_scope_first_arg", "str"}, {"get_scope_scope", "str"}, {"get_scope_thread_id", "int"}, {"get_scope_has_grad", "int"}, {"scope_struct_Load_for_Async", ""}, 
						{"FirstArgOnDemand", "str"}, 
						{"LenStrVec", "int"}, {"shuffle_str", "str"}, {"IndexStrVec", "str"}, {"str_vec_Idx", "str"}, {"str_vec_CalculateIdx", "int"}, 
						{"int_Create", "int"}, {"int_Load", "int"}, 
						{"objHash", "str"}, {"LoadObject", "str"}, {"LoadObjectScopeName", "str"}, {"object_Load_int", "int"}, {"object_Load_on_Offset_int", "int"}, 
						{"CreateNotesVector", "list"}, {"Add_To_NotesVector_float", "list"}, {"Add_To_NotesVector_int", "list"}, {"Add_To_NotesVector_str", "list"}, 
						{"list_New", "list"}, {"list_Create", "list"}, {"list_size", "int"}, {"list_CalculateIdx", "int"}, {"to_int", "int"}, {"list_CalculateSliceIdx", ""}, {"list_Slice", "list"}, 
						{"dictionary_Create", "dict"}, 

	};
}