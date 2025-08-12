

#include <map>
#include <string>
#include <vector>

#include "../compiler_frontend/include.h"




void set_functions_args_type() {


	
		
		Function_Arg_Types["LenStrVec"]["0"] = "Scope_Struct";
		Function_Arg_Types["LenStrVec"]["1"] = "str_vec";
		
		Function_Arg_Names["LenStrVec"].push_back("0");
		Function_Arg_Names["LenStrVec"].push_back("1");
		
		Function_Arg_Types["ShuffleStrVec"]["0"] = "Scope_Struct";
		Function_Arg_Types["ShuffleStrVec"]["1"] = "str_vec";
		
		Function_Arg_Names["ShuffleStrVec"].push_back("0");
		Function_Arg_Names["ShuffleStrVec"].push_back("1");
		
		Function_Arg_Types["str_vec_print"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_vec_print"]["1"] = "str_vec";
		
		Function_Arg_Names["str_vec_print"].push_back("0");
		Function_Arg_Names["str_vec_print"].push_back("1");
	
		
		
	
		
		Function_Arg_Types["_tid"]["0"] = "Scope_Struct";
		
		Function_Arg_Names["_tid"].push_back("0");
		
		Function_Arg_Types["tid"]["0"] = "Scope_Struct";
		
		Function_Arg_Names["tid"].push_back("0");
	
		
		
		
		
	
		
		Function_Arg_Types["str_str_add"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_str_add"]["1"] = "str";
		Function_Arg_Types["str_str_add"]["2"] = "str";
		
		Function_Arg_Names["str_str_add"].push_back("0");
		Function_Arg_Names["str_str_add"].push_back("1");
		Function_Arg_Names["str_str_add"].push_back("2");
		
		Function_Arg_Types["str_int_add"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_int_add"]["1"] = "str";
		Function_Arg_Types["str_int_add"]["2"] = "int";
		
		Function_Arg_Names["str_int_add"].push_back("0");
		Function_Arg_Names["str_int_add"].push_back("1");
		Function_Arg_Names["str_int_add"].push_back("2");
		
		Function_Arg_Types["str_float_add"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_float_add"]["1"] = "str";
		Function_Arg_Types["str_float_add"]["2"] = "float";
		
		Function_Arg_Names["str_float_add"].push_back("0");
		Function_Arg_Names["str_float_add"].push_back("1");
		Function_Arg_Names["str_float_add"].push_back("2");
		
		Function_Arg_Types["int_str_add"]["0"] = "Scope_Struct";
		Function_Arg_Types["int_str_add"]["1"] = "int";
		Function_Arg_Types["int_str_add"]["2"] = "str";
		
		Function_Arg_Names["int_str_add"].push_back("0");
		Function_Arg_Names["int_str_add"].push_back("1");
		Function_Arg_Names["int_str_add"].push_back("2");
		
		Function_Arg_Types["float_str_add"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_str_add"]["1"] = "float";
		Function_Arg_Types["float_str_add"]["2"] = "str";
		
		Function_Arg_Names["float_str_add"].push_back("0");
		Function_Arg_Names["float_str_add"].push_back("1");
		Function_Arg_Names["float_str_add"].push_back("2");
		
		Function_Arg_Types["SplitString"]["0"] = "Scope_Struct";
		Function_Arg_Types["SplitString"]["1"] = "str";
		Function_Arg_Types["SplitString"]["2"] = "str";
		
		Function_Arg_Names["SplitString"].push_back("0");
		Function_Arg_Names["SplitString"].push_back("1");
		Function_Arg_Names["SplitString"].push_back("2");
		
		Function_Arg_Types["str_split_idx"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_split_idx"]["1"] = "str";
		Function_Arg_Types["str_split_idx"]["2"] = "str";
		Function_Arg_Types["str_split_idx"]["3"] = "int";
		
		Function_Arg_Names["str_split_idx"].push_back("0");
		Function_Arg_Names["str_split_idx"].push_back("1");
		Function_Arg_Names["str_split_idx"].push_back("2");
		Function_Arg_Names["str_split_idx"].push_back("3");
		
		Function_Arg_Types["str_to_float"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_to_float"]["1"] = "str";
		
		Function_Arg_Names["str_to_float"].push_back("0");
		Function_Arg_Names["str_to_float"].push_back("1");
		
		Function_Arg_Types["StrToFloat"]["0"] = "Scope_Struct";
		Function_Arg_Types["StrToFloat"]["1"] = "str";
		
		Function_Arg_Names["StrToFloat"].push_back("0");
		Function_Arg_Names["StrToFloat"].push_back("1");
	
		
		Function_Arg_Types["__slee_p_"]["0"] = "Scope_Struct";
		Function_Arg_Types["__slee_p_"]["1"] = "int";
		
		Function_Arg_Names["__slee_p_"].push_back("0");
		Function_Arg_Names["__slee_p_"].push_back("1");
		
		Function_Arg_Types["random_sleep"]["0"] = "Scope_Struct";
		Function_Arg_Types["random_sleep"]["1"] = "int";
		Function_Arg_Types["random_sleep"]["2"] = "int";
		
		Function_Arg_Names["random_sleep"].push_back("0");
		Function_Arg_Names["random_sleep"].push_back("1");
		Function_Arg_Names["random_sleep"].push_back("2");
		
		Function_Arg_Types["silent_sleep"]["0"] = "Scope_Struct";
		Function_Arg_Types["silent_sleep"]["1"] = "int";
		
		Function_Arg_Names["silent_sleep"].push_back("0");
		Function_Arg_Names["silent_sleep"].push_back("1");
		
		Function_Arg_Types["start_timer"]["0"] = "Scope_Struct";
		
		Function_Arg_Names["start_timer"].push_back("0");
		
		Function_Arg_Types["end_timer"]["0"] = "Scope_Struct";
		
		Function_Arg_Names["end_timer"].push_back("0");
	
		
		
	
		
		Function_Arg_Types["nsk_vec_size"]["0"] = "Scope_Struct";
		Function_Arg_Types["nsk_vec_size"]["1"] = "Nsk_Vector";
		
		Function_Arg_Names["nsk_vec_size"].push_back("0");
		Function_Arg_Names["nsk_vec_size"].push_back("1");
	
		
		Function_Arg_Types["_quit_"]["0"] = "Scope_Struct";
		
		Function_Arg_Names["_quit_"].push_back("0");
	
		
		Function_Arg_Types["arange_int"]["0"] = "Scope_Struct";
		Function_Arg_Types["arange_int"]["1"] = "int";
		Function_Arg_Types["arange_int"]["2"] = "int";
		
		Function_Arg_Names["arange_int"].push_back("0");
		Function_Arg_Names["arange_int"].push_back("1");
		Function_Arg_Names["arange_int"].push_back("2");
		
		Function_Arg_Types["zeros_int"]["0"] = "Scope_Struct";
		Function_Arg_Types["zeros_int"]["1"] = "int";
		
		Function_Arg_Names["zeros_int"].push_back("0");
		Function_Arg_Names["zeros_int"].push_back("1");
		
		Function_Arg_Types["ones_int"]["0"] = "Scope_Struct";
		Function_Arg_Types["ones_int"]["1"] = "int";
		
		Function_Arg_Names["ones_int"].push_back("0");
		Function_Arg_Names["ones_int"].push_back("1");
		
		Function_Arg_Types["int_vec_first_nonzero"]["0"] = "Scope_Struct";
		Function_Arg_Types["int_vec_first_nonzero"]["1"] = "int_vec";
		
		Function_Arg_Names["int_vec_first_nonzero"].push_back("0");
		Function_Arg_Names["int_vec_first_nonzero"].push_back("1");
		
		Function_Arg_Types["int_vec_print"]["0"] = "Scope_Struct";
		Function_Arg_Types["int_vec_print"]["1"] = "int_vec";
		
		Function_Arg_Names["int_vec_print"].push_back("0");
		Function_Arg_Names["int_vec_print"].push_back("1");
		
		Function_Arg_Types["int_vec_size"]["0"] = "Scope_Struct";
		Function_Arg_Types["int_vec_size"]["1"] = "int_vec";
		
		Function_Arg_Names["int_vec_size"].push_back("0");
		Function_Arg_Names["int_vec_size"].push_back("1");
	
		
		Function_Arg_Types["dict_Store_Key"]["0"] = "Scope_Struct";
		Function_Arg_Types["dict_Store_Key"]["1"] = "dict";
		Function_Arg_Types["dict_Store_Key"]["2"] = "str";
		Function_Arg_Types["dict_Store_Key"]["3"] = "void";
		Function_Arg_Types["dict_Store_Key"]["4"] = "str";
		
		Function_Arg_Names["dict_Store_Key"].push_back("0");
		Function_Arg_Names["dict_Store_Key"].push_back("1");
		Function_Arg_Names["dict_Store_Key"].push_back("2");
		Function_Arg_Names["dict_Store_Key"].push_back("3");
		Function_Arg_Names["dict_Store_Key"].push_back("4");
		
		Function_Arg_Types["dict_Store_Key_int"]["0"] = "Scope_Struct";
		Function_Arg_Types["dict_Store_Key_int"]["1"] = "dict";
		Function_Arg_Types["dict_Store_Key_int"]["2"] = "str";
		Function_Arg_Types["dict_Store_Key_int"]["3"] = "int";
		
		Function_Arg_Names["dict_Store_Key_int"].push_back("0");
		Function_Arg_Names["dict_Store_Key_int"].push_back("1");
		Function_Arg_Names["dict_Store_Key_int"].push_back("2");
		Function_Arg_Names["dict_Store_Key_int"].push_back("3");
		
		Function_Arg_Types["dict_Store_Key_float"]["0"] = "Scope_Struct";
		Function_Arg_Types["dict_Store_Key_float"]["1"] = "dict";
		Function_Arg_Types["dict_Store_Key_float"]["2"] = "str";
		Function_Arg_Types["dict_Store_Key_float"]["3"] = "float";
		
		Function_Arg_Names["dict_Store_Key_float"].push_back("0");
		Function_Arg_Names["dict_Store_Key_float"].push_back("1");
		Function_Arg_Names["dict_Store_Key_float"].push_back("2");
		Function_Arg_Names["dict_Store_Key_float"].push_back("3");
		
		Function_Arg_Types["dict_print"]["0"] = "Scope_Struct";
		Function_Arg_Types["dict_print"]["1"] = "dict";
		
		Function_Arg_Names["dict_print"].push_back("0");
		Function_Arg_Names["dict_print"].push_back("1");
		
		Function_Arg_Types["dict_Query"]["0"] = "Scope_Struct";
		Function_Arg_Types["dict_Query"]["1"] = "dict";
		Function_Arg_Types["dict_Query"]["2"] = "str";
		
		Function_Arg_Names["dict_Query"].push_back("0");
		Function_Arg_Names["dict_Query"].push_back("1");
		Function_Arg_Names["dict_Query"].push_back("2");
	
		
		Function_Arg_Types["list_print"]["0"] = "Scope_Struct";
		Function_Arg_Types["list_print"]["1"] = "unknown_list";
		
		Function_Arg_Names["list_print"].push_back("0");
		Function_Arg_Names["list_print"].push_back("1");
		
		Function_Arg_Types["list_size"]["0"] = "Scope_Struct";
		Function_Arg_Types["list_size"]["1"] = "unknown_list";
		
		Function_Arg_Names["list_size"].push_back("0");
		Function_Arg_Names["list_size"].push_back("1");
		
		Function_Arg_Types["to_int"]["0"] = "Scope_Struct";
		Function_Arg_Types["to_int"]["1"] = "void";
		
		Function_Arg_Names["to_int"].push_back("0");
		Function_Arg_Names["to_int"].push_back("1");
		
		Function_Arg_Types["to_float"]["0"] = "Scope_Struct";
		Function_Arg_Types["to_float"]["1"] = "void";
		
		Function_Arg_Names["to_float"].push_back("0");
		Function_Arg_Names["to_float"].push_back("1");
	
		
		Function_Arg_Types["arange_float"]["0"] = "Scope_Struct";
		Function_Arg_Types["arange_float"]["1"] = "int";
		Function_Arg_Types["arange_float"]["2"] = "int";
		
		Function_Arg_Names["arange_float"].push_back("0");
		Function_Arg_Names["arange_float"].push_back("1");
		Function_Arg_Names["arange_float"].push_back("2");
		
		Function_Arg_Types["zeros_float"]["0"] = "Scope_Struct";
		Function_Arg_Types["zeros_float"]["1"] = "int";
		
		Function_Arg_Names["zeros_float"].push_back("0");
		Function_Arg_Names["zeros_float"].push_back("1");
		
		Function_Arg_Types["ones_float"]["0"] = "Scope_Struct";
		Function_Arg_Types["ones_float"]["1"] = "int";
		
		Function_Arg_Names["ones_float"].push_back("0");
		Function_Arg_Names["ones_float"].push_back("1");
		
		Function_Arg_Types["float_vec_first_nonzero"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_vec_first_nonzero"]["1"] = "float_vec";
		
		Function_Arg_Names["float_vec_first_nonzero"].push_back("0");
		Function_Arg_Names["float_vec_first_nonzero"].push_back("1");
		
		Function_Arg_Types["float_vec_print"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_vec_print"]["1"] = "float_vec";
		
		Function_Arg_Names["float_vec_print"].push_back("0");
		Function_Arg_Names["float_vec_print"].push_back("1");
		
		Function_Arg_Types["float_vec_size"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_vec_size"]["1"] = "float_vec";
		
		Function_Arg_Names["float_vec_size"].push_back("0");
		Function_Arg_Names["float_vec_size"].push_back("1");
	
		
		Function_Arg_Types["print"]["0"] = "Scope_Struct";
		Function_Arg_Types["print"]["1"] = "str";
		
		Function_Arg_Names["print"].push_back("0");
		Function_Arg_Names["print"].push_back("1");

}