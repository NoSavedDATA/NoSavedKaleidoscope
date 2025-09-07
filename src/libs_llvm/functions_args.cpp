

#include <map>
#include <string>
#include <vector>

#include "../compiler_frontend/include.h"




void set_functions_args_type() {


	
		
		Function_Arg_Types["LenStrVec"]["0"] = "Scope_Struct";
		Function_Arg_Types["LenStrVec"]["1"] = "str_vec";
		
		Function_Arg_DataTypes["LenStrVec"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree LenStrVec_1 = Data_Tree("vec");
		LenStrVec_1.Nested_Data.push_back(Data_Tree("str"));
		Function_Arg_DataTypes["LenStrVec"]["1"] = LenStrVec_1;
		
		Function_Arg_Names["LenStrVec"].push_back("0");
		Function_Arg_Names["LenStrVec"].push_back("1");
		
		Function_Arg_Types["ShuffleStrVec"]["0"] = "Scope_Struct";
		Function_Arg_Types["ShuffleStrVec"]["1"] = "str_vec";
		
		Function_Arg_DataTypes["ShuffleStrVec"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree ShuffleStrVec_1 = Data_Tree("vec");
		ShuffleStrVec_1.Nested_Data.push_back(Data_Tree("str"));
		Function_Arg_DataTypes["ShuffleStrVec"]["1"] = ShuffleStrVec_1;
		
		Function_Arg_Names["ShuffleStrVec"].push_back("0");
		Function_Arg_Names["ShuffleStrVec"].push_back("1");
		
		Function_Arg_Types["str_vec_print"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_vec_print"]["1"] = "str_vec";
		
		Function_Arg_DataTypes["str_vec_print"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree str_vec_print_1 = Data_Tree("vec");
		str_vec_print_1.Nested_Data.push_back(Data_Tree("str"));
		Function_Arg_DataTypes["str_vec_print"]["1"] = str_vec_print_1;
		
		Function_Arg_Names["str_vec_print"].push_back("0");
		Function_Arg_Names["str_vec_print"].push_back("1");
	
		
		
		
	
		
		Function_Arg_Types["_tid"]["0"] = "Scope_Struct";
		
		Function_Arg_DataTypes["_tid"]["0"] = Data_Tree("Scope_Struct");
		
		Function_Arg_Names["_tid"].push_back("0");
		
		Function_Arg_Types["tid"]["0"] = "Scope_Struct";
		
		Function_Arg_DataTypes["tid"]["0"] = Data_Tree("Scope_Struct");
		
		Function_Arg_Names["tid"].push_back("0");
	
		
		
		
		
		
		
	
		
		Function_Arg_Types["str_str_add"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_str_add"]["1"] = "str";
		Function_Arg_Types["str_str_add"]["2"] = "str";
		
		Function_Arg_DataTypes["str_str_add"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["str_str_add"]["1"] = Data_Tree("str");
		Function_Arg_DataTypes["str_str_add"]["2"] = Data_Tree("str");
		
		Function_Arg_Names["str_str_add"].push_back("0");
		Function_Arg_Names["str_str_add"].push_back("1");
		Function_Arg_Names["str_str_add"].push_back("2");
		
		Function_Arg_Types["str_int_add"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_int_add"]["1"] = "str";
		Function_Arg_Types["str_int_add"]["2"] = "int";
		
		Function_Arg_DataTypes["str_int_add"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["str_int_add"]["1"] = Data_Tree("str");
		Function_Arg_DataTypes["str_int_add"]["2"] = Data_Tree("int");
		
		Function_Arg_Names["str_int_add"].push_back("0");
		Function_Arg_Names["str_int_add"].push_back("1");
		Function_Arg_Names["str_int_add"].push_back("2");
		
		Function_Arg_Types["str_float_add"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_float_add"]["1"] = "str";
		Function_Arg_Types["str_float_add"]["2"] = "float";
		
		Function_Arg_DataTypes["str_float_add"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["str_float_add"]["1"] = Data_Tree("str");
		Function_Arg_DataTypes["str_float_add"]["2"] = Data_Tree("float");
		
		Function_Arg_Names["str_float_add"].push_back("0");
		Function_Arg_Names["str_float_add"].push_back("1");
		Function_Arg_Names["str_float_add"].push_back("2");
		
		Function_Arg_Types["int_str_add"]["0"] = "Scope_Struct";
		Function_Arg_Types["int_str_add"]["1"] = "int";
		Function_Arg_Types["int_str_add"]["2"] = "str";
		
		Function_Arg_DataTypes["int_str_add"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["int_str_add"]["1"] = Data_Tree("int");
		Function_Arg_DataTypes["int_str_add"]["2"] = Data_Tree("str");
		
		Function_Arg_Names["int_str_add"].push_back("0");
		Function_Arg_Names["int_str_add"].push_back("1");
		Function_Arg_Names["int_str_add"].push_back("2");
		
		Function_Arg_Types["float_str_add"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_str_add"]["1"] = "float";
		Function_Arg_Types["float_str_add"]["2"] = "str";
		
		Function_Arg_DataTypes["float_str_add"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["float_str_add"]["1"] = Data_Tree("float");
		Function_Arg_DataTypes["float_str_add"]["2"] = Data_Tree("str");
		
		Function_Arg_Names["float_str_add"].push_back("0");
		Function_Arg_Names["float_str_add"].push_back("1");
		Function_Arg_Names["float_str_add"].push_back("2");
		
		Function_Arg_Types["str_bool_add"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_bool_add"]["1"] = "str";
		Function_Arg_Types["str_bool_add"]["2"] = "bool";
		
		Function_Arg_DataTypes["str_bool_add"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["str_bool_add"]["1"] = Data_Tree("str");
		Function_Arg_DataTypes["str_bool_add"]["2"] = Data_Tree("bool");
		
		Function_Arg_Names["str_bool_add"].push_back("0");
		Function_Arg_Names["str_bool_add"].push_back("1");
		Function_Arg_Names["str_bool_add"].push_back("2");
		
		Function_Arg_Types["bool_str_add"]["0"] = "Scope_Struct";
		Function_Arg_Types["bool_str_add"]["1"] = "bool";
		Function_Arg_Types["bool_str_add"]["2"] = "str";
		
		Function_Arg_DataTypes["bool_str_add"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["bool_str_add"]["1"] = Data_Tree("bool");
		Function_Arg_DataTypes["bool_str_add"]["2"] = Data_Tree("str");
		
		Function_Arg_Names["bool_str_add"].push_back("0");
		Function_Arg_Names["bool_str_add"].push_back("1");
		Function_Arg_Names["bool_str_add"].push_back("2");
		
		Function_Arg_Types["str_split_idx"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_split_idx"]["1"] = "str";
		Function_Arg_Types["str_split_idx"]["2"] = "str";
		Function_Arg_Types["str_split_idx"]["3"] = "int";
		
		Function_Arg_DataTypes["str_split_idx"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["str_split_idx"]["1"] = Data_Tree("str");
		Function_Arg_DataTypes["str_split_idx"]["2"] = Data_Tree("str");
		Function_Arg_DataTypes["str_split_idx"]["3"] = Data_Tree("int");
		
		Function_Arg_Names["str_split_idx"].push_back("0");
		Function_Arg_Names["str_split_idx"].push_back("1");
		Function_Arg_Names["str_split_idx"].push_back("2");
		Function_Arg_Names["str_split_idx"].push_back("3");
		
		Function_Arg_Types["str_to_float"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_to_float"]["1"] = "str";
		
		Function_Arg_DataTypes["str_to_float"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["str_to_float"]["1"] = Data_Tree("str");
		
		Function_Arg_Names["str_to_float"].push_back("0");
		Function_Arg_Names["str_to_float"].push_back("1");
	
		
		Function_Arg_Types["__slee_p_"]["0"] = "Scope_Struct";
		Function_Arg_Types["__slee_p_"]["1"] = "int";
		
		Function_Arg_DataTypes["__slee_p_"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["__slee_p_"]["1"] = Data_Tree("int");
		
		Function_Arg_Names["__slee_p_"].push_back("0");
		Function_Arg_Names["__slee_p_"].push_back("1");
		
		Function_Arg_Types["random_sleep"]["0"] = "Scope_Struct";
		Function_Arg_Types["random_sleep"]["1"] = "int";
		Function_Arg_Types["random_sleep"]["2"] = "int";
		
		Function_Arg_DataTypes["random_sleep"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["random_sleep"]["1"] = Data_Tree("int");
		Function_Arg_DataTypes["random_sleep"]["2"] = Data_Tree("int");
		
		Function_Arg_Names["random_sleep"].push_back("0");
		Function_Arg_Names["random_sleep"].push_back("1");
		Function_Arg_Names["random_sleep"].push_back("2");
		
		Function_Arg_Types["silent_sleep"]["0"] = "Scope_Struct";
		Function_Arg_Types["silent_sleep"]["1"] = "int";
		
		Function_Arg_DataTypes["silent_sleep"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["silent_sleep"]["1"] = Data_Tree("int");
		
		Function_Arg_Names["silent_sleep"].push_back("0");
		Function_Arg_Names["silent_sleep"].push_back("1");
		
		Function_Arg_Types["start_timer"]["0"] = "Scope_Struct";
		
		Function_Arg_DataTypes["start_timer"]["0"] = Data_Tree("Scope_Struct");
		
		Function_Arg_Names["start_timer"].push_back("0");
		
		Function_Arg_Types["end_timer"]["0"] = "Scope_Struct";
		
		Function_Arg_DataTypes["end_timer"]["0"] = Data_Tree("Scope_Struct");
		
		Function_Arg_Names["end_timer"].push_back("0");
	
		
		
		
	
		
		Function_Arg_Types["nsk_vec_size"]["0"] = "Scope_Struct";
		Function_Arg_Types["nsk_vec_size"]["1"] = "Nsk_Vector";
		
		Function_Arg_DataTypes["nsk_vec_size"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["nsk_vec_size"]["1"] = Data_Tree("Nsk_Vector");
		
		Function_Arg_Names["nsk_vec_size"].push_back("0");
		Function_Arg_Names["nsk_vec_size"].push_back("1");
	
		
		Function_Arg_Types["str_channel_message"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_channel_message"]["1"] = "void";
		Function_Arg_Types["str_channel_message"]["2"] = "Channel";
		
		Function_Arg_DataTypes["str_channel_message"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["str_channel_message"]["1"] = Data_Tree("void");
		Function_Arg_DataTypes["str_channel_message"]["2"] = Data_Tree("Channel");
		
		Function_Arg_Names["str_channel_message"].push_back("0");
		Function_Arg_Names["str_channel_message"].push_back("1");
		Function_Arg_Names["str_channel_message"].push_back("2");
		
		Function_Arg_Types["channel_str_message"]["0"] = "Scope_Struct";
		Function_Arg_Types["channel_str_message"]["1"] = "Channel";
		Function_Arg_Types["channel_str_message"]["2"] = "str";
		
		Function_Arg_DataTypes["channel_str_message"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["channel_str_message"]["1"] = Data_Tree("Channel");
		Function_Arg_DataTypes["channel_str_message"]["2"] = Data_Tree("str");
		
		Function_Arg_Names["channel_str_message"].push_back("0");
		Function_Arg_Names["channel_str_message"].push_back("1");
		Function_Arg_Names["channel_str_message"].push_back("2");
		
		Function_Arg_Types["str_channel_terminate"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_channel_terminate"]["1"] = "Channel";
		
		Function_Arg_DataTypes["str_channel_terminate"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["str_channel_terminate"]["1"] = Data_Tree("Channel");
		
		Function_Arg_Names["str_channel_terminate"].push_back("0");
		Function_Arg_Names["str_channel_terminate"].push_back("1");
		
		Function_Arg_Types["str_channel_alive"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_channel_alive"]["1"] = "Channel";
		
		Function_Arg_DataTypes["str_channel_alive"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["str_channel_alive"]["1"] = Data_Tree("Channel");
		
		Function_Arg_Names["str_channel_alive"].push_back("0");
		Function_Arg_Names["str_channel_alive"].push_back("1");
		
		Function_Arg_Types["float_channel_message"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_channel_message"]["1"] = "void";
		Function_Arg_Types["float_channel_message"]["2"] = "Channel";
		
		Function_Arg_DataTypes["float_channel_message"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["float_channel_message"]["1"] = Data_Tree("void");
		Function_Arg_DataTypes["float_channel_message"]["2"] = Data_Tree("Channel");
		
		Function_Arg_Names["float_channel_message"].push_back("0");
		Function_Arg_Names["float_channel_message"].push_back("1");
		Function_Arg_Names["float_channel_message"].push_back("2");
		
		Function_Arg_Types["channel_float_message"]["0"] = "Scope_Struct";
		Function_Arg_Types["channel_float_message"]["1"] = "Channel";
		Function_Arg_Types["channel_float_message"]["2"] = "float";
		
		Function_Arg_DataTypes["channel_float_message"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["channel_float_message"]["1"] = Data_Tree("Channel");
		Function_Arg_DataTypes["channel_float_message"]["2"] = Data_Tree("float");
		
		Function_Arg_Names["channel_float_message"].push_back("0");
		Function_Arg_Names["channel_float_message"].push_back("1");
		Function_Arg_Names["channel_float_message"].push_back("2");
		
		Function_Arg_Types["float_channel_sum"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_channel_sum"]["1"] = "Channel";
		
		Function_Arg_DataTypes["float_channel_sum"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["float_channel_sum"]["1"] = Data_Tree("Channel");
		
		Function_Arg_Names["float_channel_sum"].push_back("0");
		Function_Arg_Names["float_channel_sum"].push_back("1");
		
		Function_Arg_Types["float_channel_mean"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_channel_mean"]["1"] = "Channel";
		
		Function_Arg_DataTypes["float_channel_mean"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["float_channel_mean"]["1"] = Data_Tree("Channel");
		
		Function_Arg_Names["float_channel_mean"].push_back("0");
		Function_Arg_Names["float_channel_mean"].push_back("1");
		
		Function_Arg_Types["float_channel_terminate"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_channel_terminate"]["1"] = "Channel";
		
		Function_Arg_DataTypes["float_channel_terminate"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["float_channel_terminate"]["1"] = Data_Tree("Channel");
		
		Function_Arg_Names["float_channel_terminate"].push_back("0");
		Function_Arg_Names["float_channel_terminate"].push_back("1");
		
		Function_Arg_Types["float_channel_alive"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_channel_alive"]["1"] = "Channel";
		
		Function_Arg_DataTypes["float_channel_alive"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["float_channel_alive"]["1"] = Data_Tree("Channel");
		
		Function_Arg_Names["float_channel_alive"].push_back("0");
		Function_Arg_Names["float_channel_alive"].push_back("1");
		
		Function_Arg_Types["int_channel_message"]["0"] = "Scope_Struct";
		Function_Arg_Types["int_channel_message"]["1"] = "void";
		Function_Arg_Types["int_channel_message"]["2"] = "Channel";
		
		Function_Arg_DataTypes["int_channel_message"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["int_channel_message"]["1"] = Data_Tree("void");
		Function_Arg_DataTypes["int_channel_message"]["2"] = Data_Tree("Channel");
		
		Function_Arg_Names["int_channel_message"].push_back("0");
		Function_Arg_Names["int_channel_message"].push_back("1");
		Function_Arg_Names["int_channel_message"].push_back("2");
		
		Function_Arg_Types["channel_int_message"]["0"] = "Scope_Struct";
		Function_Arg_Types["channel_int_message"]["1"] = "Channel";
		Function_Arg_Types["channel_int_message"]["2"] = "int";
		
		Function_Arg_DataTypes["channel_int_message"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["channel_int_message"]["1"] = Data_Tree("Channel");
		Function_Arg_DataTypes["channel_int_message"]["2"] = Data_Tree("int");
		
		Function_Arg_Names["channel_int_message"].push_back("0");
		Function_Arg_Names["channel_int_message"].push_back("1");
		Function_Arg_Names["channel_int_message"].push_back("2");
		
		Function_Arg_Types["int_channel_sum"]["0"] = "Scope_Struct";
		Function_Arg_Types["int_channel_sum"]["1"] = "Channel";
		
		Function_Arg_DataTypes["int_channel_sum"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["int_channel_sum"]["1"] = Data_Tree("Channel");
		
		Function_Arg_Names["int_channel_sum"].push_back("0");
		Function_Arg_Names["int_channel_sum"].push_back("1");
		
		Function_Arg_Types["int_channel_mean"]["0"] = "Scope_Struct";
		Function_Arg_Types["int_channel_mean"]["1"] = "Channel";
		
		Function_Arg_DataTypes["int_channel_mean"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["int_channel_mean"]["1"] = Data_Tree("Channel");
		
		Function_Arg_Names["int_channel_mean"].push_back("0");
		Function_Arg_Names["int_channel_mean"].push_back("1");
		
		Function_Arg_Types["int_channel_terminate"]["0"] = "Scope_Struct";
		Function_Arg_Types["int_channel_terminate"]["1"] = "Channel";
		
		Function_Arg_DataTypes["int_channel_terminate"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["int_channel_terminate"]["1"] = Data_Tree("Channel");
		
		Function_Arg_Names["int_channel_terminate"].push_back("0");
		Function_Arg_Names["int_channel_terminate"].push_back("1");
		
		Function_Arg_Types["int_channel_alive"]["0"] = "Scope_Struct";
		Function_Arg_Types["int_channel_alive"]["1"] = "Channel";
		
		Function_Arg_DataTypes["int_channel_alive"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["int_channel_alive"]["1"] = Data_Tree("Channel");
		
		Function_Arg_Names["int_channel_alive"].push_back("0");
		Function_Arg_Names["int_channel_alive"].push_back("1");
	
		
		Function_Arg_Types["randint"]["0"] = "Scope_Struct";
		Function_Arg_Types["randint"]["1"] = "int";
		Function_Arg_Types["randint"]["2"] = "int";
		
		Function_Arg_DataTypes["randint"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["randint"]["1"] = Data_Tree("int");
		Function_Arg_DataTypes["randint"]["2"] = Data_Tree("int");
		
		Function_Arg_Names["randint"].push_back("0");
		Function_Arg_Names["randint"].push_back("1");
		Function_Arg_Names["randint"].push_back("2");
	
		
		Function_Arg_Types["_quit_"]["0"] = "Scope_Struct";
		
		Function_Arg_DataTypes["_quit_"]["0"] = Data_Tree("Scope_Struct");
		
		Function_Arg_Names["_quit_"].push_back("0");
	
		
		Function_Arg_Types["arange_int"]["0"] = "Scope_Struct";
		Function_Arg_Types["arange_int"]["1"] = "int";
		Function_Arg_Types["arange_int"]["2"] = "int";
		
		Function_Arg_DataTypes["arange_int"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["arange_int"]["1"] = Data_Tree("int");
		Function_Arg_DataTypes["arange_int"]["2"] = Data_Tree("int");
		
		Function_Arg_Names["arange_int"].push_back("0");
		Function_Arg_Names["arange_int"].push_back("1");
		Function_Arg_Names["arange_int"].push_back("2");
		
		Function_Arg_Types["zeros_int"]["0"] = "Scope_Struct";
		Function_Arg_Types["zeros_int"]["1"] = "int";
		
		Function_Arg_DataTypes["zeros_int"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["zeros_int"]["1"] = Data_Tree("int");
		
		Function_Arg_Names["zeros_int"].push_back("0");
		Function_Arg_Names["zeros_int"].push_back("1");
		
		Function_Arg_Types["ones_int"]["0"] = "Scope_Struct";
		Function_Arg_Types["ones_int"]["1"] = "int";
		
		Function_Arg_DataTypes["ones_int"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["ones_int"]["1"] = Data_Tree("int");
		
		Function_Arg_Names["ones_int"].push_back("0");
		Function_Arg_Names["ones_int"].push_back("1");
		
		Function_Arg_Types["int_vec_first_nonzero"]["0"] = "Scope_Struct";
		Function_Arg_Types["int_vec_first_nonzero"]["1"] = "int_vec";
		
		Function_Arg_DataTypes["int_vec_first_nonzero"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree int_vec_first_nonzero_1 = Data_Tree("vec");
		int_vec_first_nonzero_1.Nested_Data.push_back(Data_Tree("int"));
		Function_Arg_DataTypes["int_vec_first_nonzero"]["1"] = int_vec_first_nonzero_1;
		
		Function_Arg_Names["int_vec_first_nonzero"].push_back("0");
		Function_Arg_Names["int_vec_first_nonzero"].push_back("1");
		
		Function_Arg_Types["int_vec_print"]["0"] = "Scope_Struct";
		Function_Arg_Types["int_vec_print"]["1"] = "int_vec";
		
		Function_Arg_DataTypes["int_vec_print"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree int_vec_print_1 = Data_Tree("vec");
		int_vec_print_1.Nested_Data.push_back(Data_Tree("int"));
		Function_Arg_DataTypes["int_vec_print"]["1"] = int_vec_print_1;
		
		Function_Arg_Names["int_vec_print"].push_back("0");
		Function_Arg_Names["int_vec_print"].push_back("1");
		
		Function_Arg_Types["int_vec_size"]["0"] = "Scope_Struct";
		Function_Arg_Types["int_vec_size"]["1"] = "int_vec";
		
		Function_Arg_DataTypes["int_vec_size"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree int_vec_size_1 = Data_Tree("vec");
		int_vec_size_1.Nested_Data.push_back(Data_Tree("int"));
		Function_Arg_DataTypes["int_vec_size"]["1"] = int_vec_size_1;
		
		Function_Arg_Names["int_vec_size"].push_back("0");
		Function_Arg_Names["int_vec_size"].push_back("1");
	
		
		Function_Arg_Types["dict_Store_Key"]["0"] = "Scope_Struct";
		Function_Arg_Types["dict_Store_Key"]["1"] = "dict";
		Function_Arg_Types["dict_Store_Key"]["2"] = "str";
		Function_Arg_Types["dict_Store_Key"]["3"] = "void";
		Function_Arg_Types["dict_Store_Key"]["4"] = "str";
		
		Function_Arg_DataTypes["dict_Store_Key"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["dict_Store_Key"]["1"] = Data_Tree("dict");
		Function_Arg_DataTypes["dict_Store_Key"]["2"] = Data_Tree("str");
		Function_Arg_DataTypes["dict_Store_Key"]["3"] = Data_Tree("void");
		Function_Arg_DataTypes["dict_Store_Key"]["4"] = Data_Tree("str");
		
		Function_Arg_Names["dict_Store_Key"].push_back("0");
		Function_Arg_Names["dict_Store_Key"].push_back("1");
		Function_Arg_Names["dict_Store_Key"].push_back("2");
		Function_Arg_Names["dict_Store_Key"].push_back("3");
		Function_Arg_Names["dict_Store_Key"].push_back("4");
		
		Function_Arg_Types["dict_Store_Key_int"]["0"] = "Scope_Struct";
		Function_Arg_Types["dict_Store_Key_int"]["1"] = "dict";
		Function_Arg_Types["dict_Store_Key_int"]["2"] = "str";
		Function_Arg_Types["dict_Store_Key_int"]["3"] = "int";
		
		Function_Arg_DataTypes["dict_Store_Key_int"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["dict_Store_Key_int"]["1"] = Data_Tree("dict");
		Function_Arg_DataTypes["dict_Store_Key_int"]["2"] = Data_Tree("str");
		Function_Arg_DataTypes["dict_Store_Key_int"]["3"] = Data_Tree("int");
		
		Function_Arg_Names["dict_Store_Key_int"].push_back("0");
		Function_Arg_Names["dict_Store_Key_int"].push_back("1");
		Function_Arg_Names["dict_Store_Key_int"].push_back("2");
		Function_Arg_Names["dict_Store_Key_int"].push_back("3");
		
		Function_Arg_Types["dict_Store_Key_float"]["0"] = "Scope_Struct";
		Function_Arg_Types["dict_Store_Key_float"]["1"] = "dict";
		Function_Arg_Types["dict_Store_Key_float"]["2"] = "str";
		Function_Arg_Types["dict_Store_Key_float"]["3"] = "float";
		
		Function_Arg_DataTypes["dict_Store_Key_float"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["dict_Store_Key_float"]["1"] = Data_Tree("dict");
		Function_Arg_DataTypes["dict_Store_Key_float"]["2"] = Data_Tree("str");
		Function_Arg_DataTypes["dict_Store_Key_float"]["3"] = Data_Tree("float");
		
		Function_Arg_Names["dict_Store_Key_float"].push_back("0");
		Function_Arg_Names["dict_Store_Key_float"].push_back("1");
		Function_Arg_Names["dict_Store_Key_float"].push_back("2");
		Function_Arg_Names["dict_Store_Key_float"].push_back("3");
		
		Function_Arg_Types["dict_print"]["0"] = "Scope_Struct";
		Function_Arg_Types["dict_print"]["1"] = "dict";
		
		Function_Arg_DataTypes["dict_print"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["dict_print"]["1"] = Data_Tree("dict");
		
		Function_Arg_Names["dict_print"].push_back("0");
		Function_Arg_Names["dict_print"].push_back("1");
		
		Function_Arg_Types["dict_Query"]["0"] = "Scope_Struct";
		Function_Arg_Types["dict_Query"]["1"] = "dict";
		Function_Arg_Types["dict_Query"]["2"] = "str";
		
		Function_Arg_DataTypes["dict_Query"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["dict_Query"]["1"] = Data_Tree("dict");
		Function_Arg_DataTypes["dict_Query"]["2"] = Data_Tree("str");
		
		Function_Arg_Names["dict_Query"].push_back("0");
		Function_Arg_Names["dict_Query"].push_back("1");
		Function_Arg_Names["dict_Query"].push_back("2");
	
		
		Function_Arg_Types["list_append_int"]["0"] = "Scope_Struct";
		Function_Arg_Types["list_append_int"]["1"] = "unknown_list";
		Function_Arg_Types["list_append_int"]["2"] = "int";
		
		Function_Arg_DataTypes["list_append_int"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["list_append_int"]["1"] = Data_Tree("unknown_list");
		Function_Arg_DataTypes["list_append_int"]["2"] = Data_Tree("int");
		
		Function_Arg_Names["list_append_int"].push_back("0");
		Function_Arg_Names["list_append_int"].push_back("1");
		Function_Arg_Names["list_append_int"].push_back("2");
		
		Function_Arg_Types["list_append_float"]["0"] = "Scope_Struct";
		Function_Arg_Types["list_append_float"]["1"] = "unknown_list";
		Function_Arg_Types["list_append_float"]["2"] = "float";
		
		Function_Arg_DataTypes["list_append_float"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["list_append_float"]["1"] = Data_Tree("unknown_list");
		Function_Arg_DataTypes["list_append_float"]["2"] = Data_Tree("float");
		
		Function_Arg_Names["list_append_float"].push_back("0");
		Function_Arg_Names["list_append_float"].push_back("1");
		Function_Arg_Names["list_append_float"].push_back("2");
		
		Function_Arg_Types["list_append_bool"]["0"] = "Scope_Struct";
		Function_Arg_Types["list_append_bool"]["1"] = "unknown_list";
		Function_Arg_Types["list_append_bool"]["2"] = "bool";
		
		Function_Arg_DataTypes["list_append_bool"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["list_append_bool"]["1"] = Data_Tree("unknown_list");
		Function_Arg_DataTypes["list_append_bool"]["2"] = Data_Tree("bool");
		
		Function_Arg_Names["list_append_bool"].push_back("0");
		Function_Arg_Names["list_append_bool"].push_back("1");
		Function_Arg_Names["list_append_bool"].push_back("2");
		
		Function_Arg_Types["list_append"]["0"] = "Scope_Struct";
		Function_Arg_Types["list_append"]["1"] = "unknown_list";
		Function_Arg_Types["list_append"]["2"] = "void";
		Function_Arg_Types["list_append"]["3"] = "str";
		
		Function_Arg_DataTypes["list_append"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["list_append"]["1"] = Data_Tree("unknown_list");
		Function_Arg_DataTypes["list_append"]["2"] = Data_Tree("void");
		Function_Arg_DataTypes["list_append"]["3"] = Data_Tree("str");
		
		Function_Arg_Names["list_append"].push_back("0");
		Function_Arg_Names["list_append"].push_back("1");
		Function_Arg_Names["list_append"].push_back("2");
		Function_Arg_Names["list_append"].push_back("3");
		
		Function_Arg_Types["list_print"]["0"] = "Scope_Struct";
		Function_Arg_Types["list_print"]["1"] = "unknown_list";
		
		Function_Arg_DataTypes["list_print"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["list_print"]["1"] = Data_Tree("unknown_list");
		
		Function_Arg_Names["list_print"].push_back("0");
		Function_Arg_Names["list_print"].push_back("1");
		
		Function_Arg_Types["list_size"]["0"] = "Scope_Struct";
		Function_Arg_Types["list_size"]["1"] = "unknown_list";
		
		Function_Arg_DataTypes["list_size"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["list_size"]["1"] = Data_Tree("unknown_list");
		
		Function_Arg_Names["list_size"].push_back("0");
		Function_Arg_Names["list_size"].push_back("1");
		
		Function_Arg_Types["to_int"]["0"] = "Scope_Struct";
		Function_Arg_Types["to_int"]["1"] = "void";
		
		Function_Arg_DataTypes["to_int"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["to_int"]["1"] = Data_Tree("void");
		
		Function_Arg_Names["to_int"].push_back("0");
		Function_Arg_Names["to_int"].push_back("1");
		
		Function_Arg_Types["to_float"]["0"] = "Scope_Struct";
		Function_Arg_Types["to_float"]["1"] = "void";
		
		Function_Arg_DataTypes["to_float"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["to_float"]["1"] = Data_Tree("void");
		
		Function_Arg_Names["to_float"].push_back("0");
		Function_Arg_Names["to_float"].push_back("1");
	
		
		Function_Arg_Types["arange_float"]["0"] = "Scope_Struct";
		Function_Arg_Types["arange_float"]["1"] = "int";
		Function_Arg_Types["arange_float"]["2"] = "int";
		
		Function_Arg_DataTypes["arange_float"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["arange_float"]["1"] = Data_Tree("int");
		Function_Arg_DataTypes["arange_float"]["2"] = Data_Tree("int");
		
		Function_Arg_Names["arange_float"].push_back("0");
		Function_Arg_Names["arange_float"].push_back("1");
		Function_Arg_Names["arange_float"].push_back("2");
		
		Function_Arg_Types["zeros_float"]["0"] = "Scope_Struct";
		Function_Arg_Types["zeros_float"]["1"] = "int";
		
		Function_Arg_DataTypes["zeros_float"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["zeros_float"]["1"] = Data_Tree("int");
		
		Function_Arg_Names["zeros_float"].push_back("0");
		Function_Arg_Names["zeros_float"].push_back("1");
		
		Function_Arg_Types["ones_float"]["0"] = "Scope_Struct";
		Function_Arg_Types["ones_float"]["1"] = "int";
		
		Function_Arg_DataTypes["ones_float"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["ones_float"]["1"] = Data_Tree("int");
		
		Function_Arg_Names["ones_float"].push_back("0");
		Function_Arg_Names["ones_float"].push_back("1");
		
		Function_Arg_Types["float_vec_first_nonzero"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_vec_first_nonzero"]["1"] = "float_vec";
		
		Function_Arg_DataTypes["float_vec_first_nonzero"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree float_vec_first_nonzero_1 = Data_Tree("vec");
		float_vec_first_nonzero_1.Nested_Data.push_back(Data_Tree("float"));
		Function_Arg_DataTypes["float_vec_first_nonzero"]["1"] = float_vec_first_nonzero_1;
		
		Function_Arg_Names["float_vec_first_nonzero"].push_back("0");
		Function_Arg_Names["float_vec_first_nonzero"].push_back("1");
		
		Function_Arg_Types["float_vec_print"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_vec_print"]["1"] = "float_vec";
		
		Function_Arg_DataTypes["float_vec_print"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree float_vec_print_1 = Data_Tree("vec");
		float_vec_print_1.Nested_Data.push_back(Data_Tree("float"));
		Function_Arg_DataTypes["float_vec_print"]["1"] = float_vec_print_1;
		
		Function_Arg_Names["float_vec_print"].push_back("0");
		Function_Arg_Names["float_vec_print"].push_back("1");
		
		Function_Arg_Types["float_vec_size"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_vec_size"]["1"] = "float_vec";
		
		Function_Arg_DataTypes["float_vec_size"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree float_vec_size_1 = Data_Tree("vec");
		float_vec_size_1.Nested_Data.push_back(Data_Tree("float"));
		Function_Arg_DataTypes["float_vec_size"]["1"] = float_vec_size_1;
		
		Function_Arg_Names["float_vec_size"].push_back("0");
		Function_Arg_Names["float_vec_size"].push_back("1");
	
		
		Function_Arg_Types["print"]["0"] = "Scope_Struct";
		Function_Arg_Types["print"]["1"] = "str";
		
		Function_Arg_DataTypes["print"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["print"]["1"] = Data_Tree("str");
		
		Function_Arg_Names["print"].push_back("0");
		Function_Arg_Names["print"].push_back("1");

}