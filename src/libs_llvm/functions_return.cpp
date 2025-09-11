

#include <map>
#include <string>
#include <vector>
#include "../data_types/data_tree.h"
#include "../compiler_frontend/include.h"


std::map<std::string, std::string> functions_return_type;

void set_functions_return_type() {

  functions_return_type = {{"gelu", "tensor"}, {"sigmoid", "tensor"}, {"_tanh", "tensor"}, {"relu", "tensor"}, {"softmax", "tensor"} , {"_glob_b_", "str_vec"}, {"glob", "str_vec"},
                           {"log", "tensor"}, {"randu_like", "tensor"}, {"RandomCrop", "tensor"}, {"RandomHorizontalFlip", "tensor"}, {"NormalizeImg", "tensor"},
                           {"dropout", "tensor"}, {"rl_discounted_return", "tensor"}, {"self_attn", "tensor"}, {"Jitter", "tensor"}, {"mse_with_priorities", "tensor"},
                           {"btc_mult", "tensor"}, {"btc_multT", "tensor"}, {"tensor_view", "tensor"}, {"clip", "tensor"}, {"tensor_argmax", "tensor"}, {"tmax", "tensor"},
                           {"tensor_onehot", "tensor"}, {"permute", "tensor"}, {"cpu", "tensor"}, {"printtt", "tensor"}, {"sum", "tensor"},
                           {"prod", "tensor"}, {"tensor_mean", "tensor"}, {"tmin", "tensor"}, {"argmin", "tensor"}, {"topk", "tensor"}, {"repeat_interleave", "tensor"},
                           {"save_img", "tensor"}, {"tensor_gpu", "tensor"}, {"save_as_int", "tensor"}, {"save_as_bin", "tensor"}, {"gather", "tensor"},
                           {"to_string", "str"}, {"cat_str_float", "str"}, {"Linear", "tensor"}, {"str_split_idx", "str"}, {"str_to_float", "float"},
                           {"mean_tensor", "tensor"},
                           {"BatchNorm2d", "tensor"}, {"Pool2d", "tensor"}, {"LSTM", "tensor"}, {"MHSA", "tensor"}, {"Embedding", "tensor"},
						{"IndexStrVec", "str"}, {"str_vec_Idx", "str"}, {"ShuffleStrVec", "str_vec"},
						{"nsk_vec_size", "int"}, {"__idx__", "int"}, {"__sliced_idx__", "int"}, 
						{"_quit_", "float"}, 
						{"dict_Create", "dict"}, {"dict_New", "dict"}, {"dict_print", "float"}, 
						{"emerge_int", "int"}, {"emerge_float", "float"}, {"_tid", "float"}, {"tid", "int"}, 
						{"str_Create", "str"}, {"str_Store", "float"}, {"str_Copy", "str"}, {"str_CopyArg", "str"}, {"str_str_add", "str"}, {"str_int_add", "str"}, {"str_float_add", "str"}, {"int_str_add", "str"}, {"float_str_add", "str"}, {"str_bool_add", "str"}, {"bool_str_add", "str"}, {"PrintStr", "float"}, {"cat_str_float", "str"}, {"str_split_idx", "str"}, {"str_to_float", "float"}, {"readline", "str"}, 
						{"bool_to_str", "str"}, 
						{"RandomStrOnDemand", "str"}, {"GetEmptyChar", "str"}, {"CopyString", "str"}, {"ConcatStr", "str"}, {"ConcatStrFreeLeft", "str"}, {"ConcatStrFreeRight", "str"}, {"ConcatStrFree", "str"}, {"ConcatFloatToStr", "str"}, {"ConcatNumToStrFree", "str"}, 
						{"silent_sleep", "float"}, {"start_timer", "float"}, {"end_timer", "float"}, 
						{"print", "float"}, 
						{"float_vec_Store_Idx", "float"}, {"arange_float", "float_vec"}, {"zeros_float", "float_vec"}, {"ones_float", "float_vec"}, {"float_vec_Idx", "float"}, {"float_vec_Idx_num", "float"}, {"float_vec_CalculateIdx", "int"}, {"float_vec_first_nonzero", "float"}, {"float_vec_print", "float"}, {"float_vec_Split_Parallel", "float_vec"}, {"float_vec_Split_Strided_Parallel", "float_vec"}, {"float_vec_size", "float"}, 
						{"int_vec_Store_Idx", "int"}, {"arange_int", "int_vec"}, {"zeros_int", "int_vec"}, {"rand_int_vec", "int_vec"}, {"ones_int", "int_vec"}, {"int_vec_Idx", "int"}, {"int_vec_Idx_num", "int"}, {"int_vec_CalculateIdx", "int"}, {"int_vec_CalculateSliceIdx", ""}, {"int_vec_Slice", "int_vec"}, {"int_vec_first_nonzero", "int"}, {"int_vec_print", "int"}, {"int_vec_Split_Parallel", "int_vec"}, {"int_vec_Split_Strided_Parallel", "int_vec"}, {"int_vec_size", "int"}, 
						{"scope_struct_Create", ""}, {"scope_struct_Copy", ""}, {"scope_struct_Overwrite", ""}, {"scope_struct_Dive", ""}, {"get_scope_first_arg", "str"}, {"get_scope_scope", "str"}, {"get_scope_thread_id", "int"}, {"get_scope_has_grad", "int"}, {"scope_struct_Reset_Threads", "float"}, {"scope_struct_Increment_Thread", "float"}, {"scope_struct_Load_for_Async", ""}, 
						{"dir_exists", "float"}, {"path_exists", "float"}, 
						{"read_float", "float"}, {"float_to_str", "str"}, 
						{"min", "float"}, {"max", "float"}, {"logE2f", "float"}, {"roundE", "float"}, {"floorE", "float"}, {"logical_not", "float"}, 
						{"FirstArgOnDemand", "str"}, 
						{"LenStrVec", "int"}, {"ShuffleStrVec", "str_vec"}, {"shuffle_str", "str"}, {"IndexStrVec", "str"}, {"str_vec_Idx", "str"}, {"str_vec_CalculateIdx", "int"}, {"str_vec_print", "float"}, 
						{"read_int", "int"}, {"int_to_str", "str"}, 
						{"objHash", "str"}, {"LoadObject", "str"}, {"InitObjectVecWithNull", "float"}, {"is_null", "float"}, {"append", "float"}, {"LoadObjectScopeName", "str"}, {"object_Load_float", "float"}, {"object_Load_int", "int"}, {"object_Load_on_Offset_float", "float"}, {"object_Load_on_Offset_int", "int"}, 
						{"print_randoms", "float"}, {"randint", "int"}, 
						{"CreateNotesVector", "list"}, {"Dispose_NotesVector", "float"}, {"Add_To_NotesVector_float", "list"}, {"Add_To_NotesVector_int", "list"}, {"Add_To_NotesVector_str", "list"}, 
						{"channel_str_message", "float"}, {"str_channel_Idx", "str"}, {"str_channel_alive", "int"}, {"float_channel_message", "float"}, {"channel_float_message", "float"}, {"float_channel_Idx", "float"}, {"float_channel_sum", "float"}, {"float_channel_mean", "float"}, {"float_channel_terminate", "float"}, {"float_channel_alive", "int"}, {"int_channel_message", "int"}, {"channel_int_message", "float"}, {"int_channel_Idx", "int"}, {"int_channel_sum", "int"}, {"int_channel_mean", "float"}, {"int_channel_terminate", "float"}, {"int_channel_alive", "bool"}, 
						{"list_New", "list"}, {"list_append", "float"}, {"list_print", "float"}, {"list_Create", "list"}, {"list_size", "int"}, {"list_CalculateIdx", "int"}, {"to_int", "int"}, {"to_float", "float"}, {"list_CalculateSliceIdx", ""}, {"list_Slice", "list"}, {"int_list_Store_Idx", "float"}, {"float_list_Store_Idx", "float"}, 

	};

	functions_return_data_type = {						{"read_float", Data_Tree("float")}, {"float_to_str", Data_Tree("str")}, 
						{"RandomStrOnDemand", Data_Tree("str")}, {"GetEmptyChar", Data_Tree("str")}, {"CopyString", Data_Tree("str")}, {"ConcatStr", Data_Tree("str")}, {"ConcatStrFreeLeft", Data_Tree("str")}, {"ConcatStrFreeRight", Data_Tree("str")}, {"ConcatStrFree", Data_Tree("str")}, {"ConcatFloatToStr", Data_Tree("str")}, {"ConcatNumToStrFree", Data_Tree("str")}, 
						{"silent_sleep", Data_Tree("float")}, {"start_timer", Data_Tree("float")}, {"end_timer", Data_Tree("float")}, 
						{"dir_exists", Data_Tree("float")}, {"path_exists", Data_Tree("float")}, 
						{"print_randoms", Data_Tree("float")}, {"randint", Data_Tree("int")}, 
						{"bool_to_str", Data_Tree("str")}, 
						{"int_vec_Store_Idx", Data_Tree("int")}, {"arange_int", Data_Tree("int_vec")}, {"zeros_int", Data_Tree("int_vec")}, {"rand_int_vec", Data_Tree("int_vec")}, {"ones_int", Data_Tree("int_vec")}, {"int_vec_Idx", Data_Tree("int")}, {"int_vec_Idx_num", Data_Tree("int")}, {"int_vec_CalculateIdx", Data_Tree("int")}, {"int_vec_CalculateSliceIdx", Data_Tree("")}, {"int_vec_Slice", Data_Tree("int_vec")}, {"int_vec_first_nonzero", Data_Tree("int")}, {"int_vec_print", Data_Tree("int")}, {"int_vec_Split_Parallel", Data_Tree("int_vec")}, {"int_vec_Split_Strided_Parallel", Data_Tree("int_vec")}, {"int_vec_size", Data_Tree("int")}, 
						{"LenStrVec", Data_Tree("int")}, {"ShuffleStrVec", Data_Tree("str_vec")}, {"shuffle_str", Data_Tree("str")}, {"IndexStrVec", Data_Tree("str")}, {"str_vec_Idx", Data_Tree("str")}, {"str_vec_CalculateIdx", Data_Tree("int")}, {"str_vec_print", Data_Tree("float")}, 
						{"FirstArgOnDemand", Data_Tree("str")}, 
						{"dict_Create", Data_Tree("dict")}, {"dict_New", Data_Tree("dict")}, {"dict_print", Data_Tree("float")}, 
						{"read_int", Data_Tree("int")}, {"int_to_str", Data_Tree("str")}, 
						{"_quit_", Data_Tree("float")}, 
						{"CreateNotesVector", Data_Tree("list")}, {"Dispose_NotesVector", Data_Tree("float")}, {"Add_To_NotesVector_float", Data_Tree("list")}, {"Add_To_NotesVector_int", Data_Tree("list")}, {"Add_To_NotesVector_str", Data_Tree("list")}, 
						{"float_vec_Store_Idx", Data_Tree("float")}, {"arange_float", Data_Tree("float_vec")}, {"zeros_float", Data_Tree("float_vec")}, {"ones_float", Data_Tree("float_vec")}, {"float_vec_Idx", Data_Tree("float")}, {"float_vec_Idx_num", Data_Tree("float")}, {"float_vec_CalculateIdx", Data_Tree("int")}, {"float_vec_first_nonzero", Data_Tree("float")}, {"float_vec_print", Data_Tree("float")}, {"float_vec_Split_Parallel", Data_Tree("float_vec")}, {"float_vec_Split_Strided_Parallel", Data_Tree("float_vec")}, {"float_vec_size", Data_Tree("float")}, 
						{"str_Create", Data_Tree("str")}, {"str_Store", Data_Tree("float")}, {"str_Copy", Data_Tree("str")}, {"str_CopyArg", Data_Tree("str")}, {"str_str_add", Data_Tree("str")}, {"str_int_add", Data_Tree("str")}, {"str_float_add", Data_Tree("str")}, {"int_str_add", Data_Tree("str")}, {"float_str_add", Data_Tree("str")}, {"str_bool_add", Data_Tree("str")}, {"bool_str_add", Data_Tree("str")}, {"PrintStr", Data_Tree("float")}, {"cat_str_float", Data_Tree("str")}, {"str_split_idx", Data_Tree("str")}, {"str_to_float", Data_Tree("float")}, {"readline", Data_Tree("str")}, 
						{"emerge_int", Data_Tree("int")}, {"emerge_float", Data_Tree("float")}, {"_tid", Data_Tree("float")}, {"tid", Data_Tree("int")}, 
						{"scope_struct_Create", Data_Tree("")}, {"scope_struct_Copy", Data_Tree("")}, {"scope_struct_Overwrite", Data_Tree("")}, {"scope_struct_Dive", Data_Tree("")}, {"get_scope_first_arg", Data_Tree("str")}, {"get_scope_scope", Data_Tree("str")}, {"get_scope_thread_id", Data_Tree("int")}, {"get_scope_has_grad", Data_Tree("int")}, {"scope_struct_Reset_Threads", Data_Tree("float")}, {"scope_struct_Increment_Thread", Data_Tree("float")}, {"scope_struct_Load_for_Async", Data_Tree("")}, 
						{"nsk_vec_size", Data_Tree("int")}, {"__idx__", Data_Tree("int")}, {"__sliced_idx__", Data_Tree("int")}, 
						{"print", Data_Tree("float")}, 
						{"min", Data_Tree("float")}, {"max", Data_Tree("float")}, {"logE2f", Data_Tree("float")}, {"roundE", Data_Tree("float")}, {"floorE", Data_Tree("float")}, {"logical_not", Data_Tree("float")}, 
						{"objHash", Data_Tree("str")}, {"LoadObject", Data_Tree("str")}, {"InitObjectVecWithNull", Data_Tree("float")}, {"is_null", Data_Tree("float")}, {"append", Data_Tree("float")}, {"LoadObjectScopeName", Data_Tree("str")}, {"object_Load_float", Data_Tree("float")}, {"object_Load_int", Data_Tree("int")}, {"object_Load_on_Offset_float", Data_Tree("float")}, {"object_Load_on_Offset_int", Data_Tree("int")}, 
						{"channel_str_message", Data_Tree("float")}, {"str_channel_Idx", Data_Tree("str")}, {"str_channel_alive", Data_Tree("int")}, {"float_channel_message", Data_Tree("float")}, {"channel_float_message", Data_Tree("float")}, {"float_channel_Idx", Data_Tree("float")}, {"float_channel_sum", Data_Tree("float")}, {"float_channel_mean", Data_Tree("float")}, {"float_channel_terminate", Data_Tree("float")}, {"float_channel_alive", Data_Tree("int")}, {"int_channel_message", Data_Tree("int")}, {"channel_int_message", Data_Tree("float")}, {"int_channel_Idx", Data_Tree("int")}, {"int_channel_sum", Data_Tree("int")}, {"int_channel_mean", Data_Tree("float")}, {"int_channel_terminate", Data_Tree("float")}, {"int_channel_alive", Data_Tree("bool")}, 
						{"list_New", Data_Tree("list")}, {"list_append", Data_Tree("float")}, {"list_print", Data_Tree("float")}, {"list_Create", Data_Tree("list")}, {"list_size", Data_Tree("int")}, {"list_CalculateIdx", Data_Tree("int")}, {"to_int", Data_Tree("int")}, {"to_float", Data_Tree("float")}, {"list_CalculateSliceIdx", Data_Tree("")}, {"list_Slice", Data_Tree("list")}, {"int_list_Store_Idx", Data_Tree("float")}, {"float_list_Store_Idx", Data_Tree("float")}, 

	};
}