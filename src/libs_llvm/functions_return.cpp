

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
                           {"save_img", "tensor"}, {"tensor_gpu", "tensor"}, {"tensor_gpuw", "tensor"}, {"save_as_int", "tensor"}, {"save_as_bin", "tensor"}, {"gather", "tensor"},
                           {"to_string", "str"}, {"cat_str_float", "str"}, {"Linear", "tensor"}, {"Conv2d", "tensor"}, {"str_split_idx", "str"}, {"str_to_float", "float"},
                           {"mean_tensor", "tensor"},
                           {"BatchNorm2d", "tensor"}, {"Pool2d", "tensor"}, {"LSTM", "tensor"}, {"MHSA", "tensor"}, {"Embedding", "tensor"},
						{"IndexStrVec", "str"}, {"str_vec_Idx", "str"}, 
						{"str_Create", "str"}, {"str_Load", "str"}, {"str_Copy", "str"}, {"str_str_add", "str"}, {"str_float_add", "str"}, {"float_str_add", "str"}, {"cat_str_float", "str"}, {"SplitString", "str_vec"}, {"str_split_idx", "str"}, 
						{"float_vec_Create", "float_vec"}, {"float_vec_Load", "float_vec"}, {"zeros_vec", "float_vec"}, {"ones_vec", "float_vec"}, 
						{"tensor_Create", "tensor"}, {"tensor_Load", "tensor"}, {"gpu", "tensor"}, {"randu_like", "tensor"}, {"tensor_view", "tensor"}, {"NewVecToTensor", "tensor"}, {"zeros_like", "tensor"}, 
						{"IndexStrVec", "str"}, {"str_vec_Idx", "str"}, 
						{"objHash", "str"}, {"LoadObject", "str"}, {"LoadObjectScopeName", "str"}, 
						{"CreateNotesVector", "list"}, {"Add_Float_To_NotesVector", "list"}, {"Add_String_To_NotesVector", "list"}, 
						{"list_New", "list"}, {"list_Load", "list"}, {"list_Create", "list"}, 
						{"dictionary_Create", "dict"}, 

	};
}