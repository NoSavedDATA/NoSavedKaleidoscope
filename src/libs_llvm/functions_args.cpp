

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
	
		
		
		
		
		Function_Arg_Types["GetEmptyChar"]["0"] = "Scope_Struct";
		
		Function_Arg_DataTypes["GetEmptyChar"]["0"] = Data_Tree("Scope_Struct");
		
		Function_Arg_Names["GetEmptyChar"].push_back("0");
		
		Function_Arg_Types["CopyString"]["0"] = "Scope_Struct";
		Function_Arg_Types["CopyString"]["1"] = "str";
		
		Function_Arg_DataTypes["CopyString"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["CopyString"]["1"] = Data_Tree("str");
		
		Function_Arg_Names["CopyString"].push_back("0");
		Function_Arg_Names["CopyString"].push_back("1");
		
		Function_Arg_Types["ConcatStr"]["0"] = "Scope_Struct";
		Function_Arg_Types["ConcatStr"]["1"] = "str";
		Function_Arg_Types["ConcatStr"]["2"] = "str";
		
		Function_Arg_DataTypes["ConcatStr"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["ConcatStr"]["1"] = Data_Tree("str");
		Function_Arg_DataTypes["ConcatStr"]["2"] = Data_Tree("str");
		
		Function_Arg_Names["ConcatStr"].push_back("0");
		Function_Arg_Names["ConcatStr"].push_back("1");
		Function_Arg_Names["ConcatStr"].push_back("2");
		
		Function_Arg_Types["ConcatStrFreeLeft"]["0"] = "Scope_Struct";
		Function_Arg_Types["ConcatStrFreeLeft"]["1"] = "str";
		Function_Arg_Types["ConcatStrFreeLeft"]["2"] = "str";
		
		Function_Arg_DataTypes["ConcatStrFreeLeft"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["ConcatStrFreeLeft"]["1"] = Data_Tree("str");
		Function_Arg_DataTypes["ConcatStrFreeLeft"]["2"] = Data_Tree("str");
		
		Function_Arg_Names["ConcatStrFreeLeft"].push_back("0");
		Function_Arg_Names["ConcatStrFreeLeft"].push_back("1");
		Function_Arg_Names["ConcatStrFreeLeft"].push_back("2");
		
		Function_Arg_Types["ConcatFloatToStr"]["0"] = "Scope_Struct";
		Function_Arg_Types["ConcatFloatToStr"]["1"] = "str";
		Function_Arg_Types["ConcatFloatToStr"]["2"] = "float";
		
		Function_Arg_DataTypes["ConcatFloatToStr"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["ConcatFloatToStr"]["1"] = Data_Tree("str");
		Function_Arg_DataTypes["ConcatFloatToStr"]["2"] = Data_Tree("float");
		
		Function_Arg_Names["ConcatFloatToStr"].push_back("0");
		Function_Arg_Names["ConcatFloatToStr"].push_back("1");
		Function_Arg_Names["ConcatFloatToStr"].push_back("2");
		
		Function_Arg_Types["ConcatNumToStrFree"]["0"] = "Scope_Struct";
		Function_Arg_Types["ConcatNumToStrFree"]["1"] = "str";
		Function_Arg_Types["ConcatNumToStrFree"]["2"] = "float";
		
		Function_Arg_DataTypes["ConcatNumToStrFree"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["ConcatNumToStrFree"]["1"] = Data_Tree("str");
		Function_Arg_DataTypes["ConcatNumToStrFree"]["2"] = Data_Tree("float");
		
		Function_Arg_Names["ConcatNumToStrFree"].push_back("0");
		Function_Arg_Names["ConcatNumToStrFree"].push_back("1");
		Function_Arg_Names["ConcatNumToStrFree"].push_back("2");
	
		
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
		
		Function_Arg_Types["str_str_different"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_str_different"]["1"] = "str";
		Function_Arg_Types["str_str_different"]["2"] = "str";
		
		Function_Arg_DataTypes["str_str_different"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["str_str_different"]["1"] = Data_Tree("str");
		Function_Arg_DataTypes["str_str_different"]["2"] = Data_Tree("str");
		
		Function_Arg_Names["str_str_different"].push_back("0");
		Function_Arg_Names["str_str_different"].push_back("1");
		Function_Arg_Names["str_str_different"].push_back("2");
		
		Function_Arg_Types["str_str_equal"]["0"] = "Scope_Struct";
		Function_Arg_Types["str_str_equal"]["1"] = "str";
		Function_Arg_Types["str_str_equal"]["2"] = "str";
		
		Function_Arg_DataTypes["str_str_equal"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["str_str_equal"]["1"] = Data_Tree("str");
		Function_Arg_DataTypes["str_str_equal"]["2"] = Data_Tree("str");
		
		Function_Arg_Names["str_str_equal"].push_back("0");
		Function_Arg_Names["str_str_equal"].push_back("1");
		Function_Arg_Names["str_str_equal"].push_back("2");
		
		Function_Arg_Types["readline"]["0"] = "Scope_Struct";
		
		Function_Arg_DataTypes["readline"]["0"] = Data_Tree("Scope_Struct");
		
		Function_Arg_Names["readline"].push_back("0");
	
		
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
	
		
		
		
	
		
		Function_Arg_Types["allocate_void"]["0"] = "Scope_Struct";
		Function_Arg_Types["allocate_void"]["1"] = "int";
		Function_Arg_Types["allocate_void"]["2"] = "const";
		Function_Arg_Types["allocate_void"]["3"] = "type";
		Function_Arg_Types["allocate_void"]["4"] = "type";
		Function_Arg_Types["allocate_void"]["5"] = "";
		Function_Arg_Types["allocate_void"]["6"] = "auto";
		Function_Arg_Types["allocate_void"]["7"] = "it";
		Function_Arg_Types["allocate_void"]["8"] = "data_name_to_type";
		Function_Arg_Types["allocate_void"]["9"] = "type";
		Function_Arg_Types["allocate_void"]["10"] = "type";
		Function_Arg_Types["allocate_void"]["11"] = "if";
		Function_Arg_Types["allocate_void"]["12"] = "it";
		Function_Arg_Types["allocate_void"]["13"] = "it";
		Function_Arg_Types["allocate_void"]["14"] = "data_name_to_type";
		
		Function_Arg_DataTypes["allocate_void"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["allocate_void"]["1"] = Data_Tree("int");
		Function_Arg_DataTypes["allocate_void"]["2"] = Data_Tree("const");
		Function_Arg_DataTypes["allocate_void"]["3"] = Data_Tree("type");
		Function_Arg_DataTypes["allocate_void"]["4"] = Data_Tree("type");
		Function_Arg_DataTypes["allocate_void"]["5"] = Data_Tree("");
		Function_Arg_DataTypes["allocate_void"]["6"] = Data_Tree("auto");
		Function_Arg_DataTypes["allocate_void"]["7"] = Data_Tree("it");
		Function_Arg_DataTypes["allocate_void"]["8"] = Data_Tree("data_name_to_type");
		Function_Arg_DataTypes["allocate_void"]["9"] = Data_Tree("type");
		Function_Arg_DataTypes["allocate_void"]["10"] = Data_Tree("type");
		Function_Arg_DataTypes["allocate_void"]["11"] = Data_Tree("if");
		Function_Arg_DataTypes["allocate_void"]["12"] = Data_Tree("it");
		Function_Arg_DataTypes["allocate_void"]["13"] = Data_Tree("it");
		Function_Arg_DataTypes["allocate_void"]["14"] = Data_Tree("data_name_to_type");
		
		Function_Arg_Names["allocate_void"].push_back("0");
		Function_Arg_Names["allocate_void"].push_back("1");
		Function_Arg_Names["allocate_void"].push_back("2");
		Function_Arg_Names["allocate_void"].push_back("3");
		Function_Arg_Names["allocate_void"].push_back("4");
		Function_Arg_Names["allocate_void"].push_back("5");
		Function_Arg_Names["allocate_void"].push_back("6");
		Function_Arg_Names["allocate_void"].push_back("7");
		Function_Arg_Names["allocate_void"].push_back("8");
		Function_Arg_Names["allocate_void"].push_back("9");
		Function_Arg_Names["allocate_void"].push_back("10");
		Function_Arg_Names["allocate_void"].push_back("11");
		Function_Arg_Names["allocate_void"].push_back("12");
		Function_Arg_Names["allocate_void"].push_back("13");
		Function_Arg_Names["allocate_void"].push_back("14");
	
		
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
		
		Function_Arg_Types["rand_int_vec"]["0"] = "Scope_Struct";
		Function_Arg_Types["rand_int_vec"]["1"] = "int";
		Function_Arg_Types["rand_int_vec"]["2"] = "int";
		Function_Arg_Types["rand_int_vec"]["3"] = "int";
		
		Function_Arg_DataTypes["rand_int_vec"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["rand_int_vec"]["1"] = Data_Tree("int");
		Function_Arg_DataTypes["rand_int_vec"]["2"] = Data_Tree("int");
		Function_Arg_DataTypes["rand_int_vec"]["3"] = Data_Tree("int");
		
		Function_Arg_Names["rand_int_vec"].push_back("0");
		Function_Arg_Names["rand_int_vec"].push_back("1");
		Function_Arg_Names["rand_int_vec"].push_back("2");
		Function_Arg_Names["rand_int_vec"].push_back("3");
		
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
	
		
		Function_Arg_Types["read_int"]["0"] = "Scope_Struct";
		
		Function_Arg_DataTypes["read_int"]["0"] = Data_Tree("Scope_Struct");
		
		Function_Arg_Names["read_int"].push_back("0");
		
		Function_Arg_Types["int_to_str"]["0"] = "Scope_Struct";
		Function_Arg_Types["int_to_str"]["1"] = "int";
		
		Function_Arg_DataTypes["int_to_str"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["int_to_str"]["1"] = Data_Tree("int");
		
		Function_Arg_Names["int_to_str"].push_back("0");
		Function_Arg_Names["int_to_str"].push_back("1");
	
		
		Function_Arg_Types["list_append_int"]["0"] = "Scope_Struct";
		Function_Arg_Types["list_append_int"]["1"] = "list";
		Function_Arg_Types["list_append_int"]["2"] = "int";
		
		Function_Arg_DataTypes["list_append_int"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree list_append_int_1 = Data_Tree("list");
		list_append_int_1.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["list_append_int"]["1"] = list_append_int_1;
		Function_Arg_DataTypes["list_append_int"]["2"] = Data_Tree("int");
		
		Function_Arg_Names["list_append_int"].push_back("0");
		Function_Arg_Names["list_append_int"].push_back("1");
		Function_Arg_Names["list_append_int"].push_back("2");
		
		Function_Arg_Types["list_append_float"]["0"] = "Scope_Struct";
		Function_Arg_Types["list_append_float"]["1"] = "list";
		Function_Arg_Types["list_append_float"]["2"] = "float";
		
		Function_Arg_DataTypes["list_append_float"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree list_append_float_1 = Data_Tree("list");
		list_append_float_1.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["list_append_float"]["1"] = list_append_float_1;
		Function_Arg_DataTypes["list_append_float"]["2"] = Data_Tree("float");
		
		Function_Arg_Names["list_append_float"].push_back("0");
		Function_Arg_Names["list_append_float"].push_back("1");
		Function_Arg_Names["list_append_float"].push_back("2");
		
		Function_Arg_Types["list_append_bool"]["0"] = "Scope_Struct";
		Function_Arg_Types["list_append_bool"]["1"] = "list";
		Function_Arg_Types["list_append_bool"]["2"] = "bool";
		
		Function_Arg_DataTypes["list_append_bool"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree list_append_bool_1 = Data_Tree("list");
		list_append_bool_1.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["list_append_bool"]["1"] = list_append_bool_1;
		Function_Arg_DataTypes["list_append_bool"]["2"] = Data_Tree("bool");
		
		Function_Arg_Names["list_append_bool"].push_back("0");
		Function_Arg_Names["list_append_bool"].push_back("1");
		Function_Arg_Names["list_append_bool"].push_back("2");
		
		Function_Arg_Types["list_append"]["0"] = "Scope_Struct";
		Function_Arg_Types["list_append"]["1"] = "list";
		Function_Arg_Types["list_append"]["2"] = "void";
		Function_Arg_Types["list_append"]["3"] = "str";
		
		Function_Arg_DataTypes["list_append"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree list_append_1 = Data_Tree("list");
		list_append_1.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["list_append"]["1"] = list_append_1;
		Function_Arg_DataTypes["list_append"]["2"] = Data_Tree("void");
		Function_Arg_DataTypes["list_append"]["3"] = Data_Tree("str");
		
		Function_Arg_Names["list_append"].push_back("0");
		Function_Arg_Names["list_append"].push_back("1");
		Function_Arg_Names["list_append"].push_back("2");
		Function_Arg_Names["list_append"].push_back("3");
		
		Function_Arg_Types["list_print"]["0"] = "Scope_Struct";
		Function_Arg_Types["list_print"]["1"] = "list";
		
		Function_Arg_DataTypes["list_print"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree list_print_1 = Data_Tree("list");
		list_print_1.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["list_print"]["1"] = list_print_1;
		
		Function_Arg_Names["list_print"].push_back("0");
		Function_Arg_Names["list_print"].push_back("1");
		
		Function_Arg_Types["tuple_print"]["0"] = "Scope_Struct";
		Function_Arg_Types["tuple_print"]["1"] = "list";
		
		Function_Arg_DataTypes["tuple_print"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree tuple_print_1 = Data_Tree("list");
		tuple_print_1.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["tuple_print"]["1"] = tuple_print_1;
		
		Function_Arg_Names["tuple_print"].push_back("0");
		Function_Arg_Names["tuple_print"].push_back("1");
		
		Function_Arg_Types["list_shuffle"]["0"] = "Scope_Struct";
		Function_Arg_Types["list_shuffle"]["1"] = "list";
		
		Function_Arg_DataTypes["list_shuffle"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree list_shuffle_1 = Data_Tree("list");
		list_shuffle_1.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["list_shuffle"]["1"] = list_shuffle_1;
		
		Function_Arg_Names["list_shuffle"].push_back("0");
		Function_Arg_Names["list_shuffle"].push_back("1");
		
		Function_Arg_Types["list_size"]["0"] = "Scope_Struct";
		Function_Arg_Types["list_size"]["1"] = "list";
		
		Function_Arg_DataTypes["list_size"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree list_size_1 = Data_Tree("list");
		list_size_1.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["list_size"]["1"] = list_size_1;
		
		Function_Arg_Names["list_size"].push_back("0");
		Function_Arg_Names["list_size"].push_back("1");
		
		Function_Arg_Types["list_as_float_vec"]["0"] = "Scope_Struct";
		Function_Arg_Types["list_as_float_vec"]["1"] = "list";
		
		Function_Arg_DataTypes["list_as_float_vec"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree list_as_float_vec_1 = Data_Tree("list");
		list_as_float_vec_1.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["list_as_float_vec"]["1"] = list_as_float_vec_1;
		
		Function_Arg_Names["list_as_float_vec"].push_back("0");
		Function_Arg_Names["list_as_float_vec"].push_back("1");
		
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
		
		Function_Arg_Types["zip"]["0"] = "Scope_Struct";
		Function_Arg_Types["zip"]["1"] = "list";
		Function_Arg_Types["zip"]["2"] = "list";
		Function_Arg_Types["zip"]["3"] = "list";
		Function_Arg_Types["zip"]["4"] = "list";
		Function_Arg_Types["zip"]["5"] = "list";
		Function_Arg_Types["zip"]["6"] = "list";
		Function_Arg_Types["zip"]["7"] = "list";
		Function_Arg_Types["zip"]["8"] = "list";
		Function_Arg_Types["zip"]["9"] = "list";
		Function_Arg_Types["zip"]["10"] = "list";
		Function_Arg_Types["zip"]["11"] = "list";
		
		Function_Arg_DataTypes["zip"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree zip_1 = Data_Tree("list");
		zip_1.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["zip"]["1"] = zip_1;
		Data_Tree zip_2 = Data_Tree("list");
		zip_2.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["zip"]["2"] = zip_2;
		Data_Tree zip_3 = Data_Tree("list");
		zip_3.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["zip"]["3"] = zip_3;
		Data_Tree zip_4 = Data_Tree("list");
		zip_4.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["zip"]["4"] = zip_4;
		Data_Tree zip_5 = Data_Tree("list");
		zip_5.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["zip"]["5"] = zip_5;
		Data_Tree zip_6 = Data_Tree("list");
		zip_6.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["zip"]["6"] = zip_6;
		Data_Tree zip_7 = Data_Tree("list");
		zip_7.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["zip"]["7"] = zip_7;
		Data_Tree zip_8 = Data_Tree("list");
		zip_8.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["zip"]["8"] = zip_8;
		Data_Tree zip_9 = Data_Tree("list");
		zip_9.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["zip"]["9"] = zip_9;
		Data_Tree zip_10 = Data_Tree("list");
		zip_10.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["zip"]["10"] = zip_10;
		Data_Tree zip_11 = Data_Tree("list");
		zip_11.Nested_Data.push_back(Data_Tree("any"));
		Function_Arg_DataTypes["zip"]["11"] = zip_11;
		
		Function_Arg_Names["zip"].push_back("0");
		Function_Arg_Names["zip"].push_back("1");
		Function_Arg_Names["zip"].push_back("2");
		Function_Arg_Names["zip"].push_back("3");
		Function_Arg_Names["zip"].push_back("4");
		Function_Arg_Names["zip"].push_back("5");
		Function_Arg_Names["zip"].push_back("6");
		Function_Arg_Names["zip"].push_back("7");
		Function_Arg_Names["zip"].push_back("8");
		Function_Arg_Names["zip"].push_back("9");
		Function_Arg_Names["zip"].push_back("10");
		Function_Arg_Names["zip"].push_back("11");
	
		
		Function_Arg_Types["read_float"]["0"] = "Scope_Struct";
		
		Function_Arg_DataTypes["read_float"]["0"] = Data_Tree("Scope_Struct");
		
		Function_Arg_Names["read_float"].push_back("0");
		
		Function_Arg_Types["float_to_str"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_to_str"]["1"] = "float";
		
		Function_Arg_DataTypes["float_to_str"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["float_to_str"]["1"] = Data_Tree("float");
		
		Function_Arg_Names["float_to_str"].push_back("0");
		Function_Arg_Names["float_to_str"].push_back("1");
		
		Function_Arg_Types["nsk_pow"]["0"] = "Scope_Struct";
		Function_Arg_Types["nsk_pow"]["1"] = "float";
		Function_Arg_Types["nsk_pow"]["2"] = "float";
		
		Function_Arg_DataTypes["nsk_pow"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["nsk_pow"]["1"] = Data_Tree("float");
		Function_Arg_DataTypes["nsk_pow"]["2"] = Data_Tree("float");
		
		Function_Arg_Names["nsk_pow"].push_back("0");
		Function_Arg_Names["nsk_pow"].push_back("1");
		Function_Arg_Names["nsk_pow"].push_back("2");
		
		Function_Arg_Types["nsk_sqrt"]["0"] = "Scope_Struct";
		Function_Arg_Types["nsk_sqrt"]["1"] = "float";
		
		Function_Arg_DataTypes["nsk_sqrt"]["0"] = Data_Tree("Scope_Struct");
		Function_Arg_DataTypes["nsk_sqrt"]["1"] = Data_Tree("float");
		
		Function_Arg_Names["nsk_sqrt"].push_back("0");
		Function_Arg_Names["nsk_sqrt"].push_back("1");
	
		
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
		
		Function_Arg_Types["float_vec_pow"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_vec_pow"]["1"] = "float_vec";
		Function_Arg_Types["float_vec_pow"]["2"] = "float";
		
		Function_Arg_DataTypes["float_vec_pow"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree float_vec_pow_1 = Data_Tree("vec");
		float_vec_pow_1.Nested_Data.push_back(Data_Tree("float"));
		Function_Arg_DataTypes["float_vec_pow"]["1"] = float_vec_pow_1;
		Function_Arg_DataTypes["float_vec_pow"]["2"] = Data_Tree("float");
		
		Function_Arg_Names["float_vec_pow"].push_back("0");
		Function_Arg_Names["float_vec_pow"].push_back("1");
		Function_Arg_Names["float_vec_pow"].push_back("2");
		
		Function_Arg_Types["float_vec_sum"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_vec_sum"]["1"] = "float_vec";
		
		Function_Arg_DataTypes["float_vec_sum"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree float_vec_sum_1 = Data_Tree("vec");
		float_vec_sum_1.Nested_Data.push_back(Data_Tree("float"));
		Function_Arg_DataTypes["float_vec_sum"]["1"] = float_vec_sum_1;
		
		Function_Arg_Names["float_vec_sum"].push_back("0");
		Function_Arg_Names["float_vec_sum"].push_back("1");
		
		Function_Arg_Types["float_vec_int_add"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_vec_int_add"]["1"] = "float_vec";
		Function_Arg_Types["float_vec_int_add"]["2"] = "int";
		
		Function_Arg_DataTypes["float_vec_int_add"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree float_vec_int_add_1 = Data_Tree("vec");
		float_vec_int_add_1.Nested_Data.push_back(Data_Tree("float"));
		Function_Arg_DataTypes["float_vec_int_add"]["1"] = float_vec_int_add_1;
		Function_Arg_DataTypes["float_vec_int_add"]["2"] = Data_Tree("int");
		
		Function_Arg_Names["float_vec_int_add"].push_back("0");
		Function_Arg_Names["float_vec_int_add"].push_back("1");
		Function_Arg_Names["float_vec_int_add"].push_back("2");
		
		Function_Arg_Types["float_vec_int_div"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_vec_int_div"]["1"] = "float_vec";
		Function_Arg_Types["float_vec_int_div"]["2"] = "int";
		
		Function_Arg_DataTypes["float_vec_int_div"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree float_vec_int_div_1 = Data_Tree("vec");
		float_vec_int_div_1.Nested_Data.push_back(Data_Tree("float"));
		Function_Arg_DataTypes["float_vec_int_div"]["1"] = float_vec_int_div_1;
		Function_Arg_DataTypes["float_vec_int_div"]["2"] = Data_Tree("int");
		
		Function_Arg_Names["float_vec_int_div"].push_back("0");
		Function_Arg_Names["float_vec_int_div"].push_back("1");
		Function_Arg_Names["float_vec_int_div"].push_back("2");
		
		Function_Arg_Types["float_vec_float_vec_add"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_vec_float_vec_add"]["1"] = "float_vec";
		Function_Arg_Types["float_vec_float_vec_add"]["2"] = "float_vec";
		
		Function_Arg_DataTypes["float_vec_float_vec_add"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree float_vec_float_vec_add_1 = Data_Tree("vec");
		float_vec_float_vec_add_1.Nested_Data.push_back(Data_Tree("float"));
		Function_Arg_DataTypes["float_vec_float_vec_add"]["1"] = float_vec_float_vec_add_1;
		Data_Tree float_vec_float_vec_add_2 = Data_Tree("vec");
		float_vec_float_vec_add_2.Nested_Data.push_back(Data_Tree("float"));
		Function_Arg_DataTypes["float_vec_float_vec_add"]["2"] = float_vec_float_vec_add_2;
		
		Function_Arg_Names["float_vec_float_vec_add"].push_back("0");
		Function_Arg_Names["float_vec_float_vec_add"].push_back("1");
		Function_Arg_Names["float_vec_float_vec_add"].push_back("2");
		
		Function_Arg_Types["float_vec_float_vec_sub"]["0"] = "Scope_Struct";
		Function_Arg_Types["float_vec_float_vec_sub"]["1"] = "float_vec";
		Function_Arg_Types["float_vec_float_vec_sub"]["2"] = "float_vec";
		
		Function_Arg_DataTypes["float_vec_float_vec_sub"]["0"] = Data_Tree("Scope_Struct");
		Data_Tree float_vec_float_vec_sub_1 = Data_Tree("vec");
		float_vec_float_vec_sub_1.Nested_Data.push_back(Data_Tree("float"));
		Function_Arg_DataTypes["float_vec_float_vec_sub"]["1"] = float_vec_float_vec_sub_1;
		Data_Tree float_vec_float_vec_sub_2 = Data_Tree("vec");
		float_vec_float_vec_sub_2.Nested_Data.push_back(Data_Tree("float"));
		Function_Arg_DataTypes["float_vec_float_vec_sub"]["2"] = float_vec_float_vec_sub_2;
		
		Function_Arg_Names["float_vec_float_vec_sub"].push_back("0");
		Function_Arg_Names["float_vec_float_vec_sub"].push_back("1");
		Function_Arg_Names["float_vec_float_vec_sub"].push_back("2");
		
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