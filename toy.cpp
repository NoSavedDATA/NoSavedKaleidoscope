
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "src/KaleidoscopeJIT.h"




#include <algorithm>
#include <cstdarg>
#include <cassert>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>
#include <iomanip>
#include <math.h>
#include <fenv.h>
#include <tuple>
#include <chrono>
#include <thread>
#include <random>
#include <float.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>






#include "src/include.h"






using namespace llvm;
using namespace llvm::orc;








std::map<std::string, int> NotatorsMap = {
  {"bias", bias},
  {"fp32", fp32},
  {"fp16", fp16},
  {"causal", causal},
};





LCG rng(generate_custom_seed());






  // Error Colors
// \033[0m default
// \033[31m red
// \033[33m yellow
// \033[34m blue
// \033[95m purple











//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//

//global




// Vars
std::map<std::string, std::vector<char *>> ClassStrVecs;
std::map<std::string, DT_int_vec *> NamedIntVecs;
std::map<std::string, float> NamedClassValues;
std::map<std::string, int> NamedInts;
std::map<std::string, std::vector<std::pair<std::string, std::string>>> ScopeVarsToClean;
std::map<std::string, char *> ScopeNamesToClean;
std::map<int, std::map<std::string, std::vector<std::string>>> ThreadedScopeTensorsToClean;






// File Handling
std::vector<char *> glob_str_files;




// Handle Class self with phantom argument






//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//


  

static void HandleImport() {
    Parser_Struct parser_struct;
    parser_struct.line = LineCounter;
    ParseImport(parser_struct);
}

static void HandleClass() {
    Parser_Struct parser_struct;
    parser_struct.line = LineCounter;
    ParseClass(parser_struct);
}

static void HandleDefinition() {
  
  Parser_Struct parser_struct;
  if (auto FnAST = ParseDefinition(parser_struct)) {

    FunctionProtos[FnAST->getProto().getName()] =
      std::make_unique<PrototypeAST>(FnAST->getProto());

    ExitOnErr(TheJIT->addAST(std::move(FnAST)));
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern() {
  Parser_Struct parser_struct;
  if (auto ProtoAST = ParseExtern(parser_struct)) {
    if (auto *FnIR = ProtoAST->codegen()) {
      fprintf(stderr, "Read extern: ");
      FnIR->print(errs());
      fprintf(stderr, "\n");
      FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

std::vector<std::thread> all_threads;

static void CodegenTopLevelExpression(std::unique_ptr<FunctionAST> &FnAST) {

    auto *FnIR =  FnAST->codegen();

    /*
    fprintf(stderr, "\nRead top-level expression:");
    FnIR->print(errs());
    fprintf(stderr, "\n\n");
    */


    // TheModule->print(llvm::errs(), nullptr);

    // Create a ResourceTracker for memory managment
    // anonymous expression -- that way we can free it after executing.
    auto RT = TheJIT->getMainJITDylib().createResourceTracker();

    auto TSM = ThreadSafeModule(std::move(TheModule), std::move(TheContext));
    ExitOnErr(TheJIT->addModule(std::move(TSM), RT));
    // Add IR module


    InitializeModule();

    // Points __anon_expr
    auto Sym = ExitOnErr(TheJIT->lookup("__anon_expr"));
    //assert(Sym && "Function not found");
      
      
    // Get the symbol's address and cast it to the right type (takes no
    // arguments, returns a float) so we can call it as a native function.
    auto *FP = Sym.getAddress().toPtr<float (*)()>();
    auto fp = FP();
    
    // fprintf(stderr, "%.2f\n", fp);

    // Delete the anonymous expression module from the JIT.
    ExitOnErr(RT->remove());    
}



static void HandleTopLevelExpression() {
  // Evaluate a top-level expression into an anonymous function.
  
  Parser_Struct parser_struct;
  parser_struct.function_name = "__anon_expr";
  if (std::unique_ptr<FunctionAST> FnAST = ParseTopLevelExpr(parser_struct)) {
    CodegenTopLevelExpression(std::ref(FnAST));

	
  
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}


void InitializeTokenizer() {

    if (Sys_Arguments.size()>0)
    {
        tokenizer.openFile(Sys_Arguments[0]);
        getNextToken();

    } else
        getNextToken();

}

/// top ::= definition | external | expression | ';'
static void MainLoop() {
    while (true) {
         // std::cout << "MAIN LOOP, reading token: " << CurTok << "/" << ReverseToken(CurTok) << "\n";
        switch (CurTok) {
        case 13:
            std::cout << "FOUND CARRIAGE RETURN" << ".\n";
            break;
        case tok_eof:
            return;
        case ';': // ignore top-level semicolons.
            getNextToken();
            break;
        case '.': 
            getNextToken();
            break;
        case tok_space:
            getNextToken();
            break;
        case tok_tab:
            getNextToken();
            break;
        case tok_def:
            HandleDefinition();
            break;
        case tok_class:
            HandleClass();
            break;
        case tok_import:
            HandleImport();
            break;
        case tok_extern:
            HandleExtern();
            break;
        case tok_constructor:
            LogErrorNextBlock(LineCounter, "Constructor has no class associated.");
            break;
        default:
            
            // std::cout << "Wait top level" <<  ".\n";
            // std::cout << "reading token: " << CurTok << "/" << ReverseToken(CurTok) << "\n";

            HandleTopLevelExpression(); 
            // std::cout << "Finished top level" <<  ".\n";
            break;
        }
    }
}


//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

/// putchard - putchar that takes a float and returns 0.
extern "C" float putchard(float X) {
  fputc((char)X, stderr);
  return 0;
}

/// printd - printf that takes a float prints it as "%f\n", returning 0.
extern "C" float printd(float X) {
  fprintf(stderr, "%f\n", X);
  return 0;
}

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//

__attribute__((constructor))
void early_init() {
    // std::cout << "Constructor Function Executed\n";
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter(); // Prepare for target hardware
  InitializeNativeTargetAsmParser();
}

int main(int argc, char* argv[]) {
    has_main=true;

    for (int i = 1; i < argc; i++) {
        // std::cout << "Argument " << i << ": " << argv[i] << "\n";
        Sys_Arguments.push_back(argv[i]);
        if(i==1) {
           tokenizer.files.pop();
           tokenizer.files.push(argv[i]);
        }
    }


  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence[tok_space] = 1;
  BinopPrecedence['$'] = 1;
  BinopPrecedence['='] = 4;
  BinopPrecedence[tok_arrow] = 4;
  BinopPrecedence['!'] = 9;
  BinopPrecedence[tok_and] = 9;
  BinopPrecedence[tok_not] = 9;
  BinopPrecedence[tok_or] = 9;
  BinopPrecedence[tok_xor] = 9;
  BinopPrecedence['>'] = 10;
  BinopPrecedence['<'] = 10;
  BinopPrecedence[tok_equal] = 10;
  BinopPrecedence[tok_diff] = 10;
  BinopPrecedence[tok_minor_eq] = 10;
  BinopPrecedence[tok_higher_eq] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['%'] = 35;
  BinopPrecedence['*'] = 39;
  BinopPrecedence['/'] = 40;
  BinopPrecedence[tok_int_div] = 40;
  BinopPrecedence['^'] = 50;
  BinopPrecedence['@'] = 60;


  floatFunctions["log"] = "logE";
  floatFunctions["log2"] = "logE2";
  floatFunctions["log2f"] = "logE2f";
  floatFunctions["round"] = "roundE";
  floatFunctions["floor"] = "floorE";


    gc_sizes[0] = 8;
    gc_sizes[1] = 16;
    gc_sizes[2] = 24;
    gc_sizes[3] = 48;
    gc_sizes[4] = 64;
    gc_sizes[5] = 128;
    gc_sizes[6] = 256;
    gc_sizes[7] = 384;
    gc_sizes[8] = 512;
    gc_sizes[9] = 768;
    gc_sizes[10] = 1024;
    gc_sizes[11] = 2048;
    gc_sizes[12] = 4096;
    gc_sizes[13] = 8192;
    gc_sizes[14] = GC_max_object_size;


    for (int i=0, c=0; i<=GC_N; i++) {
        int size = i * GC_ALIGN;
        while (c<GC_obj_sizes-1 && gc_sizes[c]<size)
            c++;
        GC_size_to_class[i] = gc_sizes[c];
    }
    for (int i=0; i<GC_obj_sizes; ++i)
        GC_span_traits_vec.emplace(gc_sizes[i], new GC_span_traits(gc_sizes[i]));
        


  set_functions_return_type();
  set_functions_args_type();
  set_user_functions();
  vararg_methods = {"tensor_view", "tensor_sum", "tensor_mean", "mean_tensor" ,"tensor_prod", "tensor_tmax", "tensor_argmax", "tensor_load_bin_idx", "zip"};


  return_tensor_functions = {"gelu", "sigmoid", "_tanh", "relu", "softmax", "log", "randu_like",
                             "RandomCrop", "RandomHorizontalFlip", "NormalizeImg", "dropout", "sigmoid_add2weights",
                             "rl_discounted_return", "self_attn", "Jitter", "mse_with_priorities",
                             "btc_mult", "btc_multT", "Linear"};

  
  

  return_tensor_fn = concat_str_vec(return_tensor_functions, return_tensor_methods);

  return_pinned_methods = {"gpu", "gpuw"};


  // Universal
  string_methods = {"split", "split_idx"};


  // tensor + string + ...
  // e.g: x.view(), str.split()
  native_methods = {"split", "split_idx", "str_vec_print"};
  native_methods = concat_str_vec(native_methods, return_tensor_methods);
  native_methods = concat_str_vec(native_methods, user_cpp_functions);

  return_string_fn = {"to_string", "cat_str_float"};


  native_functions = {"ShuffleStrVec", "gload_img", "wload_img", "silent_sleep", "__slee_p_",
                      "LenStrVec", "zeros_vec", "ones_vec", "start_timer", "end_timer",
                      "_glob_b_", "print", "cross_entropy", "backprop", "AdamW", "SGD",
                      "load_preprocess_img", "max", "min", "unbug",
                      "cpu_idx", "OneCycleLR", "CosineLR", "wload_img_resize",
                      "build_vocab", "tokenize", "wtokenize", "write_zerosw",
                      "wtokenize_pad_left", "print_randoms", "wtokenize_pad_left_batch_first",
                      "wtokenize_pad_left_idx", "print_scope", "load_bin", "wload_bin", "randint",
                      "print_tensor", "path_exists", "dir_exists", "load_bin_idx",
                      "network_ema", "mse", "priority_sample", "priority_sample_val",
                      "importance_sample_idx", "importance_sample_weight",
                      "cross_entropy_idx"};
  native_functions = concat_str_vec(native_functions, return_tensor_functions);
  native_functions = concat_str_vec(native_functions, return_string_fn);
  native_fn = concat_str_vec(native_methods, native_functions);



  reverse_ops = {{"float_tensor", "tensor_float"}};

  elements_type_return = {{"tensor_tensor", "tensor"}, {"float_float", "float"}, {"str_str", "str"}, {"str_float", "str"},
                     {"float_str", "str"}, {"int_int", "int"}, {"int_float", "float"}, {"float_int", "float"}, {"str_int", "str"}, {"int_str", "str"},
                     {"str_bool", "str"}, {"bool_str", "str"}, {"bool_bool", "bool"},
                     {"tensor_float", "tensor"}, {"pinned_tensor_pinned_tensor", "pinned_tensor"},
                     {"pinned_tensor_tensor", "pinned_tensor"}, {"pinned_tensor_float", "pinned_tensor"},
                     {"object_object", "object"}, {"str_object", "object"},
                     {"tensor_int", "tensor"}, {"int_tensor", "tensor"}, {"str_channel", "str"}, {"channel_str", "float"}, {"channel_int", "float"},
                     {"int_channel", "int"}, {"channel_float", "float"}, {"float_channel", "float"}};

  ops_type_return = {{"int_int_higher", "bool"}, {"int_int_minor", "bool"}, {"int_int_equal", "bool"}, {"int_int_different", "bool"},
                     {"int_int_higher_eq", "bool"}, {"int_int_minor_eq", "bool"},
                     {"float_float_higher", "bool"}, {"float_float_minor", "bool"}, {"float_float_equal", "bool"}, {"float_float_different", "bool"},
                     {"float_float_higher_eq", "bool"}, {"float_float_minor_eq", "bool"}};

                     

  op_map = {{'*', "mult"}, {'@', "mma"},  {'+', "add"}, {'-', "sub"}, {'/', "div"}, {'<', "minor"}, {'>', "higher"}, {tok_equal, "equal"},
            {tok_diff, "different"}, {tok_higher_eq, "higher_eq"}, {tok_minor_eq, "minor_eq"}, {'%', "mod"}, {'=', "attr"},
            {77, "error"}, {tok_arrow, "message"}, {tok_and, "and"}, {tok_not, "not"}, {tok_or, "or"}, {tok_xor, "xor"}};

  for (auto pair : op_map)
    op_map_names.push_back(pair.second);


  
  notators_str = {"bias", "fp32", "fp16", "causal"};


  // Prime the first token.


  InitializeTokenizer();

  TheJIT = ExitOnErr(KaleidoscopeJIT::Create());
  InitializeModule();


  MainLoop();

  return 0;
}

