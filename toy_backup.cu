// JIT
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

#include "include/KaleidoscopeJIT.h"


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
#include <glob.h>
#include <chrono>
#include <thread>
#include <random>
#include <float.h>
#include <fstream>
#include <sstream>
#include <filesystem>



// Cuda
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <omp.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/cutlass.h"
#include <cudnn.h>
#include <mma.h>

#include "include/cu_commons.h"


pthread_mutex_t mutex, clean_scope_mutex, char_pool_mutex, vocab_mutex, random_seed_mutex, aux_mutex;

float TERMINATE_VARARG = -40370000000.0f;
int UNK_TOK = 1.0f;
int PAD_TOK = 0.0f;

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

int WARP_SIZE;

// Files
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb/stb_image_write.h"


cudaDeviceProp deviceProp;

int THREADS_PER_BLOCK = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

const int TILE_SIZE = (int)floorf(sqrtf((float)THREADS_PER_BLOCK)); 
const int TILE_SIZE_SQ = TILE_SIZE*TILE_SIZE;


static cublasHandle_t cublas_handle;
static cublasLtHandle_t cublaslt_handle;
cudnnHandle_t cudnn;

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
static size_t cublaslt_workspace_size = 32 * 1024 * 1024; // 32 MB
static void* cublaslt_workspace = NULL;
static cublasComputeType_t cublas_compute_type;


cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;

#define CUBLAS_LOWP CUDA_R_32F
#define PRECISION_MODE PRECISION_FP32



using namespace llvm;
using namespace llvm::orc;
using namespace nvcuda;

#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#else
#define MAX_1024_THREADS_BLOCKS 1
#endif

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

// Error Colors

// \033[0m default
// \033[31m red
// \033[33m yellow
// \033[95m purple



unsigned int generate_custom_seed() {
    // Combine time, process ID, and thread ID to generate a seed
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    unsigned int nanoseconds = get_millisecond_time();

    unsigned int tid = std::hash<std::thread::id>{}(std::this_thread::get_id());


    unsigned int seed = nanoseconds ^ tid;
    return seed;
}



class PhiloxRNG {
public:
    using uint32 = uint32_t;
    using uint64 = uint64_t;

    PhiloxRNG(uint64 seed1, uint64 seed2) : key{ static_cast<uint32>(seed1), static_cast<uint32>(seed1 >> 32),
                                                 static_cast<uint32>(seed2), static_cast<uint32>(seed2 >> 32) } {}

    std::array<uint32, 4> operator()() {
        std::array<uint32, 4> ctr = { counter++, 0, 0, 0 };
        for (int i = 0; i < rounds; i++) {
            ctr = singleRound(ctr);
        }
        return ctr;
    }

private:
    uint32 counter = 0;
    uint32 key[4];
    static constexpr int rounds = 10;

    std::array<uint32, 4> singleRound(const std::array<uint32, 4>& ctr) const {
        std::array<uint32, 4> output;
        const uint64 mult = 0xD2511F53;
        const uint64 add = 0xCD9E8D57;

        uint64 x = static_cast<uint64>(ctr[0]) * mult;
        uint64 y = static_cast<uint64>(ctr[2]) * add;

        output[0] = static_cast<uint32>(y >> 32) ^ ctr[1] ^ key[0];
        output[1] = static_cast<uint32>(x >> 32) ^ ctr[3] ^ key[1];
        output[2] = static_cast<uint32>(y) ^ ctr[2];
        output[3] = static_cast<uint32>(x) ^ ctr[0];

        return output;
    }
};



class MT19937 {
public:
    MT19937(uint32_t seed) {
        // Initialize the state vector
        state[0] = seed;
        for (int i = 1; i < n; i++) {
            state[i] = f * (state[i - 1] ^ (state[i - 1] >> (w - 2))) + i;
        }
        index = n;
    }

    uint32_t extract() {
        if (index >= n) {
            twist();
        }

        uint32_t y = state[index++];
        // Tempering
        y ^= (y >> u);
        y ^= (y << s) & b;
        y ^= (y << t) & c;
        y ^= (y >> l);

        return y;
    }

private:
    static const int w = 32;
    static const int n = 624;
    static const int m = 397;
    static const int r = 31;
    static const uint32_t a = 0x9908B0DF;
    static const int u = 11;
    static const uint32_t d = 0xFFFFFFFF;
    static const int s = 7;
    static const uint32_t b = 0x9D2C5680;
    static const int t = 15;
    static const uint32_t c = 0xEFC60000;
    static const int l = 18;
    static const uint32_t f = 1812433253;

    uint32_t state[n];
    int index;

    void twist() {
        for (int i = 0; i < n; i++) {
            uint32_t x = (state[i] & 0x80000000) + (state[(i + 1) % n] & 0x7FFFFFFF);
            uint32_t xA = x >> 1;
            if (x % 2 != 0) {
                xA ^= a;
            }
            state[i] = state[(i + m) % n] ^ xA;
        }
        index = 0;
    }
};


class LCG {
public:
    LCG(uint32_t seed) : state(seed) {}

    uint32_t next() {
        state = (a * state + c) % m;
        return state;
    }

    void setSeed(uint32_t seed) {
        state = seed;
    }

private:
    uint32_t state;
    static constexpr uint32_t a = 1664525; // Multiplier
    static constexpr uint32_t c = 1013904223; // Increment
    static constexpr uint32_t m = 0xFFFFFFFF; // Modulus (2^32 - 1)
};


bool in_str(std::string str, std::vector<std::string> list);

//MT19937 mt(generate_custom_seed());
LCG rng(generate_custom_seed());

//std::random_device rd; // it is already defined at cu_common.h
std::mt19937 MAIN_PRNG(rd()^get_millisecond_time());


unsigned long long get_int_seed()
{
  std::uniform_int_distribution<unsigned long long> dist(0, ULLONG_MAX);
  return dist(MAIN_PRNG);
}

std::vector<std::string> rds;


char *RandomString(size_t length) {
  //unsigned int seed = generate_custom_seed();

  const std::string charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  pthread_mutex_lock(&random_seed_mutex);

  //MT19937 mt(generate_custom_seed());
  //LCG rng(generate_custom_seed());

  char *random_string = new char[length+1];

  for (int i = 0; i < length; i++) {

      //int random_index = mt.extract() % charset.length();
      int random_index = rng.next() % charset.length();
      random_string[i] = charset[random_index];
  }

  //random_string[length] = '\0';

  //std::cout << "" << random_string << "\n";

  pthread_mutex_unlock(&random_seed_mutex);

  
  //std::string aux = random_string;
  //if(!in_str(aux,rds))
  //  rds.push_back(aux);
  

  return random_string;
}





/*
const std::string charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
char *RandomString(size_t length) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution(0, charset.size() - 1);

  char *random_string = new char[length+1];
  for (size_t i = 0; i < length; i++) {
    int random_index = distribution(generator);
    random_string[i] = charset[random_index];
  }

  return random_string;
}
*/


/*
const std::string charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
char *RandomString(size_t length) {
    
  
  //unsigned int seed = generate_custom_seed();
  std::random_device rd;
  unsigned int rand_seed = rd();

  char *random_string = new char[length+1];

  for (size_t i = 0; i < length; i++) {
      int random_index = rand_r(&rand_seed) % charset.size();
      random_string[i] = charset[random_index];
  }

  return random_string;
}
*/



bool ends_with(std::string str_input, std::string str_end)
{
  return str_input.size() >= str_end.size() && str_input.compare(str_input.size() - str_end.size(), str_end.size(), str_end) == 0;
}
bool begins_with(const std::string& str_input, const std::string& str_start) {
    return str_input.size() >= str_start.size() && str_input.compare(0, str_start.size(), str_start) == 0;
}
bool contains_str(const std::string& str_input, const std::string& str_sub) {
    return str_input.find(str_sub) != std::string::npos;
}
std::string remove_substring(const std::string& str, const std::string& substr) {
    std::string result = str;  // Copy the original string
    size_t pos = result.find(substr);
    if (pos != std::string::npos) {
        result.erase(pos, substr.length());
    }
    return result;
}
bool starts_with(const char* str, const char* sub) {
  return strncmp(str, sub, strlen(sub)) == 0;
}

char *str_to_char(std::string str)
{
    char *c_str = new char[str.length() + 1]; // +1 for the null terminator
    std::strcpy(c_str, str.c_str());

    return c_str;
}





int count_pattern(const std::string& text, const std::string& pattern) {
  int count = 0;
  size_t pos = 0;

  std::cout << "Trying to count"  << "\n";
  // Iterate while finding occurrences of the pattern
  while ((pos = text.find(pattern, pos)) != std::string::npos) {
    count++;
    pos += pattern.length(); // Move to the character after the found pattern
  }

  return count;
}

std::vector<std::string> split_str(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream stream(str);

  while (std::getline(stream, token, delimiter)) {
    tokens.push_back(token);
  }

  return tokens;
}

std::vector<std::string> split(const char* input, const std::string& delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0, end = 0;
    while ((end = std::string(input + start).find(delimiter)) != std::string::npos) {
        tokens.push_back(std::string(input + start, end));
        start += end + delimiter.length();
    }
    tokens.push_back(std::string(input + start));
    return tokens;
}


bool in_char(char ch, const std::vector<char>& list) {
  // Use std::find to efficiently search the list for the character
  return std::find(list.begin(), list.end(), ch) != list.end();
}

bool in_str(std::string str, std::vector<std::string> list) {
    return std::find(list.begin(), list.end(), str) != list.end();
}

bool in_int(int value, const std::vector<int>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}

bool in_int_ptr(int value, int *list, int size) {
  for (int i=0; i<size; ++i)
  {
    if (list[i]==value)
      return true;
  }
  return false;
}

bool in_float_vec(float value, const std::vector<float>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}

bool in_char_ptr_vec(const char *value, const std::vector<char *>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}
bool in_float_ptr_vec(const float *value, const std::vector<float *>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}

std::vector<std::string> concat_str_vec(std::vector<std::string> l, std::vector<std::string>r)
{
  std::vector<std::string> concatenated_vectors = l;
  concatenated_vectors.insert(concatenated_vectors.end(), r.begin(), r.end());
  return concatenated_vectors;
}
std::vector<int> concat_int_vec(std::vector<int> l, std::vector<int>r)
{
  std::vector<int> concatenated_vectors = l;
  concatenated_vectors.insert(concatenated_vectors.end(), r.begin(), r.end());
  return concatenated_vectors;
}

void move_to_char_pool(size_t, char *, std::string);
char *get_from_char_pool(size_t, std::string);

// Tensor related
std::vector<std::string> return_tensor_functions, return_tensor_methods, return_tensor_fn, native_modules,
return_pinned_methods, vararg_methods, string_methods, native_methods, native_functions, native_fn, tensor_inits,
return_string_fn, threaded_tensor_functions, require_scope_functions, notators_str;






PointerType *floatPtrTy, *int8PtrTy;
bool ShallCodegen = true;

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//






// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known words.
enum Token {
  tok_eof = -1,

  // functions/classes
  tok_def = -2,
  tok_class = -77,
  tok_self = -78,
  tok_class_attr = -79,
  tok_extern = -3,

  // primary
  tok_identifier = -4,
  tok_number = -5,
  tok_str = -40, // ""

  // control
  tok_if = -6,
  tok_then = -7,
  tok_else = -8,
  tok_for = -9,
  tok_while = -10,
  tok_async = -22,
  tok_async_finish = -23,
  tok_lock = -26,
  tok_unlock = -27,
  tok_tab = 9,
  tok_return = -32,
  tok_as = -33,

  // operators
  tok_binary = -11,
  tok_unary = -12,
  tok_equal = -28,
  tok_diff = -34,
  tok_higher_eq = -35,
  tok_minor_eq = -36,
  tok_mod = -29,


  tok_space = -14,

  
  // var definition
  tok_var = -15,
  tok_tensor = -16,
  tok_param = -44,
  tok_pinned_tensor = -25,
  tok_var_str = -17,
  tok_str_vec = -24,
  tok_float_vec = -31,
  tok_attr_var = -18,
  tok_attr_tensor = -19,
  tok_conv2d = -21,
  tok_maxpool2d = -41,
  tok_avgpool2d = -42,
  tok_batchnorm2d = -43,
  tok_lstm = -47,
  tok_embedding = -48,
  tok_mhsa = -51,
  tok_linear = -52,
  tok_bn2drelu = -45,
  tok_relu = -46,
  tok_vec = -37,
  tok_post_class_attr_attr = -38,
  tok_post_class_attr_identifier = -39,

  tok_global = -49,
  tok_no_grad = -50,


};


enum Types {
  type_float = 0,
  type_tensor = 1,
  type_pinned_tensor = 2,
  type_object = 3,
  type_string = 4,
};

enum NameSolverTypes {
  type_self = 0,
  type_attr = 1,
  type_vec = 2,
  type_var = 3,
  type_object_name = 4,
  type_object_vec = 5
};

enum NN_Mode {
  eval_mode = 0,
  training_mode = 1,
};



enum BackwardTypes {
  create_tensor_from_brackets_op=-4,
  create_tensor_op=-3,
  leaf=-2,
  tensor_leaf = -1,
  weight_leaf = 0,
  bias_leaf = 1,
  attribution = 2,
  mult_op = 3,
  conv2d = 4,
  maxpool2d = 5,
  batchnorm2d = 6,
  bn2drelu = 32,
  relu_op = 7,
  cudnn_relu_op = 33,
  gelu_op = 8,
  sigmoid_op = 9,
  tanh_op = 10,
  cross_entropy_op = 11,
  cross_entropy_idx_op = 54,
  mse_op = 44,
  add_op = 12,
  sub_op = 25,
  hadamard_op = 13,
  div_op=26,
  softmax_op = 14,
  onehot_op = 15,
  view_op = 16,
  sum_op = 17,
  mean_op = 18,
  max_op = 19,
  argmax_op = 20,
  topk_op = 21,
  clip_op = 22,
  gpu_op = 23,
  cpu_op = 24,
  equal_op = 27,
  crop_op = 28,
  random_horizontal_flip_op = 29,
  normalize_img_op = 30,
  jitter_op = 46,
  randu_like_op = 31,
  scalar_add_op = 34,
  scalar_sub_op = 35,
  scalar_mult_op = 36,
  scalar_div_op = 37,
  dropout_op = 38,
  sigmoid_add2weights_op = 39,
  lstm_op = 40,
  embedding_op = 41,
  detach_op = 42,
  gather_last_dim_op = 43,
  self_attn_op = 45,
  mse_is_w_op = 47,
  no_op = 48,
  lgrad_op = 49,
  broadcast_lastdim_add_op = 50,
  idx_with_tensor_op = 51,
  mhsa_op = 52,
  mean_over_semilast_dim_op = 53,
  linear_op = 55,
};

int nn_mode=training_mode;
std::vector<int> leaf_ops, loss_ops, gradless_ops, activation_ops, preprocessing_ops, tensor_scalar_ops, custom_ops, weightless_ops;


enum Notators {
  bias=0,
  fp32=1,
  fp16=2,
  causal=3,
};

std::map<std::string, int> NotatorsMap = {
  {"bias", bias},
  {"fp32", fp32},
  {"fp16", fp16},
  {"causal", causal},
};

std::map<int, std::string> token_to_string = {
  { tok_eof, "eof" },

  // functions/classes
  { tok_def, "def" },
  { tok_class, "class" },
  { tok_self, "self" },
  { tok_class_attr, "object attr" },
  { tok_extern, "extern" },

  // primary
  { tok_identifier, "tok identifier" },
  { tok_number, "tok number" },
  { tok_str, "tok str `` ''" },
  { tok_str_vec, "tok str vector" },
  { tok_float_vec, "tok float vec" },

  

  // control
  { tok_if, "if" },
  { tok_then, "then" },
  { tok_else, "else" },
  { tok_for, "for" },
  { tok_while, "while" },
  { tok_async, "async" },
  { tok_async_finish, "finish" },
  { tok_tab, "tok tab" },
  { tok_return, "tok return"},
  { tok_as, "tok as"},
  { tok_vec, "tok vec"},


  // operators
  { tok_binary, "tok binary" },
  { tok_unary,"tok unary" },


  { tok_space, "tok_space" },

  { tok_post_class_attr_attr, ".attr."},
  { tok_post_class_attr_identifier, ".identifier"},
  
  // var definition
  { tok_var, "float"},
  { tok_tensor, "tensor"},
  { tok_var_str, "var str"},
  { tok_attr_var, "tok attr var"},
  { tok_attr_tensor, "tok attr tensor"},
  { tok_conv2d, "Conv2d"},
  { tok_lstm, "LSTM"},
  { tok_embedding, "Embedding"},
  { tok_maxpool2d, "MaxPool2d"},
  { tok_avgpool2d, "AvgPool2d"},
  { tok_batchnorm2d, "BatchNorm2d"},
  { tok_bn2drelu, "BN2dRelu"},
  { tok_relu, "Relu"},

  { tok_global, "global"},
  { tok_no_grad, "no_grad"},

  { tok_mhsa, "MHSA"},
  { tok_linear, "Linear"},
  

  { 10, "tok space"},

  
  { 40, "(" },
  { 41, ")" },

  { 42, "*" },
  { 43, "+" },
  { 44, "," },
  { 45, "-" },
  { 47, "/" },

  { 48, "0" },
  { 49, "1" },
  { 50, "2" },
  { 51, "3" },
  { 52, "4" },
  { 53, "5" },
  { 54, "6" },
  { 55, "7" },
  { 56, "8" },
  { 57, "9" },
  { 58, ":" },
  { 59, ";" },
  { 60, "<" },
  { 61, "=" },
  { 62, ">" },
  { 64, "@" },

  { 91, "[" },
  { 93, "]" },

  { tok_equal, "==" },
  { tok_diff, "!=" },
  { tok_higher_eq, ">=" },
  { tok_minor_eq, "<=" },
  { tok_mod, "//" },


  { static_cast<int>('a'), "a" },
  { static_cast<int>('b'), "b" },
  { static_cast<int>('c'), "c" },
  { static_cast<int>('d'), "d" },
  { static_cast<int>('e'), "e" },
  { static_cast<int>('f'), "f" },
  { static_cast<int>('g'), "g" },
  { static_cast<int>('h'), "h" },
  { static_cast<int>('i'), "i" },
  { static_cast<int>('j'), "j" },
  { static_cast<int>('k'), "k" },
  { static_cast<int>('l'), "l" },
  { static_cast<int>('m'), "m" },
  { static_cast<int>('n'), "n" },
  { static_cast<int>('o'), "o" },
  { static_cast<int>('p'), "p" },
  { static_cast<int>('q'), "q" },
  { static_cast<int>('r'), "r" },
  { static_cast<int>('s'), "s" },
  { static_cast<int>('t'), "t" },
  { static_cast<int>('u'), "u" },
  { static_cast<int>('v'), "v" },
  { static_cast<int>('w'), "w" },
  { static_cast<int>('x'), "x" },
  { static_cast<int>('y'), "y" },
  { static_cast<int>('z'), "z" },

};
std::vector<char> ops = {'+', '-', '*', '/', '@', '=', '>', '<', 10, -14, ',', '(', ')', ';', tok_equal, tok_diff, tok_higher_eq, tok_minor_eq};
std::vector<char> terminal_tokens = {';', tok_def, tok_extern, tok_class};


static std::string IdentifierStr; // Filled in if tok_identifier
static float NumVal;             // Filled in if tok_number

std::string ReverseToken(int _char)
{
  /*
  if (_char>=48 && _char<=57) // Handle number
    return std::to_string(NumVal);
  */
  if (_char==tok_identifier)
    return IdentifierStr;

  return token_to_string[_char];
}

int LineCounter = 1;

int SeenTabs = 0;
int LastSeenTabs = 0;

/// get_token - Return the next token from standard input.
static int get_token() {
  static int LastChar = ' ';

  

  /*
  if (LastChar!=32)
    std::cout << "Pre last char: " << ReverseToken(LastChar) << "\n";
  */

  // Skip any whitespace and backspace.
  
  
  while (LastChar==32 || LastChar==tok_tab)
    LastChar = getchar();
    
  if (LastChar=='[')
  {
    LastChar = getchar();
    return '[';
  }

  //std::cout << "Last char: " << LastChar << "\n";
    
  if (LastChar=='"')
  {

    LastChar = getchar();
    IdentifierStr = LastChar;

    bool name_ok=true;
    while (name_ok)
    {
      LastChar = getchar();
      
      if(LastChar!='"')
        IdentifierStr += LastChar;
      else
        name_ok = false;

    }
    LastChar = getchar();
    
    return tok_str;
  }

  
  if (LastChar=='.')
  {
    LastChar = getchar(); // eat .
    IdentifierStr = LastChar;
    bool name_ok=true;
    while (name_ok)
    {
      LastChar = getchar();
      
      
      if(isalnum(LastChar) || LastChar=='_')
        IdentifierStr += LastChar;
      else
        name_ok = false;

      
      if (LastChar=='.')
      {
        LastChar = getchar();
        return tok_post_class_attr_attr;
      }
    }
    
    return tok_post_class_attr_identifier;
  }

  if (isalpha(LastChar) || LastChar=='_') { // identifier: [a-zA-Z][a-zA-Z0-9]*
    IdentifierStr = LastChar;
    bool name_ok=true;
    while (name_ok)
    {
      LastChar = getchar();

      if (LastChar=='[')
        break;
      
      if(isalnum(LastChar) || LastChar=='_')
        IdentifierStr += LastChar;
      else
        name_ok = false;

      if (IdentifierStr == "tensor")
      {
        LastChar = getchar();
        return tok_tensor;
      }
      if (IdentifierStr == "param")
      {
        LastChar = getchar();
        return tok_param;
      }
      if (IdentifierStr == "pinned_tensor")
      {
        LastChar = getchar();
        return tok_pinned_tensor;
      }
      if (IdentifierStr == "Conv2d")
      {
        LastChar = getchar();
        return tok_conv2d;
      }
      if (IdentifierStr == "LSTM")
      {
        LastChar = getchar();
        return tok_lstm;
      }
      if (IdentifierStr == "MHSA")
      {
        LastChar = getchar();
        return tok_mhsa;
      }
      if (IdentifierStr == "Linear")
      {
        LastChar = getchar();
        return tok_linear;
      }
      if (IdentifierStr == "Embedding")
      {
        LastChar = getchar();
        return tok_embedding;
      }
      if (IdentifierStr == "MaxPool2d")
      {
        LastChar = getchar();
        return tok_maxpool2d;
      }
      if (IdentifierStr == "AvgPool2d")
      {
        LastChar = getchar();
        return tok_avgpool2d;
      }
      if (IdentifierStr == "BatchNorm2d")
      {
        LastChar = getchar();
        return tok_batchnorm2d;
      }
      if (IdentifierStr == "BN2dRelu")
      {
        LastChar = getchar();
        return tok_bn2drelu;
      }
      if (IdentifierStr == "Relu")
      {
        LastChar = getchar();
        return tok_relu;
      }
      if (LastChar=='.')
      {
        LastChar = getchar();
        if (IdentifierStr == "self")
          return tok_self;
        return tok_class_attr;
      }
    }

    if (IdentifierStr == "def")
      return tok_def;
    if (IdentifierStr == "class")
      return tok_class;
    if (IdentifierStr == "extern")
      return tok_extern;
    if (IdentifierStr == "if")
      return tok_if;
    if (IdentifierStr == "then")
      return tok_then;
    if (IdentifierStr == "else")
      return tok_else;
    if (IdentifierStr == "for")
      return tok_for;
    if (IdentifierStr == "while")
      return tok_while;
    if (IdentifierStr == "async")
      return tok_async;
    if (IdentifierStr == "finish")
      return tok_async_finish;
    if (IdentifierStr == "global")
      return tok_global;
    if (IdentifierStr == "no_grad")
      return tok_no_grad;
    if (IdentifierStr == "lock")
      return tok_lock;
    if (IdentifierStr == "unlock")
      return tok_unlock;
    if (IdentifierStr == "binary")
      return tok_binary;
    if (IdentifierStr == "unary")
      return tok_unary;
    if (IdentifierStr == "float")
      return tok_var;
    if (IdentifierStr == "glob")
      IdentifierStr = "_glob_b_";
    if (IdentifierStr == "tanh")
      IdentifierStr = "_tanh";
    if (IdentifierStr == "str")
      return tok_var_str;
    if (IdentifierStr == "str_vec")
      return tok_str_vec;
    if (IdentifierStr == "float_vec")
      return tok_float_vec;
    if (IdentifierStr == "return")
      return tok_return;
    if (IdentifierStr == "as")
      return tok_as;
    if (IdentifierStr == "vec")
      return tok_vec;
    return tok_identifier;
  }

  if (isdigit(LastChar) || LastChar == '.') { // Number: [-.]+[0-9.]+
    
    std::string NumStr;
    if (LastChar == '-') { // Check for optional minus sign
      NumStr += LastChar;
      LastChar = getchar();
    }
    do {
      NumStr += LastChar;
      LastChar = getchar();
    } while (isdigit(LastChar) || LastChar == '.');

    NumVal = strtod(NumStr.c_str(), nullptr);
    return tok_number;
  }

  if (LastChar == '#') {
    // Comment until end of line.
    do
      LastChar = getchar();
    while (LastChar != EOF && LastChar != '\n' && LastChar != tok_space && LastChar != '\r');

    if (LastChar != EOF)
      return get_token();
  }

  // Check for end of file.  Don't eat the EOF.
  if (LastChar == EOF)
    return tok_eof;

  // Otherwise, just return the character as its ascii value.
  int ThisChar = LastChar;


  
  if (ThisChar==10 || LastChar==tok_tab)
  {
    

    int seen_spaces=0;

    while(LastChar==10 || LastChar==tok_tab || LastChar==32) {
      if(ThisChar==10)
      {
        LineCounter += 1;
        LastSeenTabs = SeenTabs;
        SeenTabs = 0;
        seen_spaces = 0;
      }
      if (LastChar==tok_tab)
        SeenTabs+=1;
      if (LastChar==32)
        seen_spaces+=1;
      if (seen_spaces==3)
      {
        seen_spaces=0;
        SeenTabs+=1;
      }

      ThisChar = (int)LastChar;
      LastChar = getchar(); 
    }
    //std::cout << "\nThisChar: " << ThisChar << " LastChar " << LastChar << "\n";

    //std::cout << "New seen tabs: " << SeenTabs << "\n";
    return tok_space;
  }


  LastChar = getchar();
  int otherChar = LastChar;



  if (ThisChar=='=' && otherChar=='=')
  {
    LastChar = getchar();
    return tok_equal;
  }
  if (ThisChar=='!' && otherChar=='=')
  {
    LastChar = getchar();
    return tok_diff;
  }
  if (ThisChar=='>' && otherChar=='=')
  {
    LastChar = getchar();
    return tok_higher_eq;
  }
  if (ThisChar=='<' && otherChar=='=')
  {
    LastChar = getchar();
    return tok_minor_eq;
  }

  if((ThisChar=='/')&&(otherChar == '/')){
    LastChar = getchar();
    return 77;
  }

  //std::cout << "Post char: " << ReverseToken(ThisChar) << "\n";

  // else: return ascii number of the character.
  return ThisChar;
}

//===----------------------------------------------------------------------===//
// Abstract Syntax Tree (aka Parse Tree)
//===----------------------------------------------------------------------===//

/// ExprAST - Base class for all expression nodes.
class ExprAST {
public:
  virtual ~ExprAST() = default;
  std::vector<float> Dims = {-1.0f};
  std::string Type = "None";
  std::string ReturnType = "None";
  std::string Name = "Unnamed";
  bool isSelf = false;
  bool isAttribute = false;
  std::string _pre_dot = "";
  bool isVec = false;
  bool isVarLoad = false;
  bool SolverIncludeScope = true;
  bool NameSolveToLast = true;

  Value *TensorPtr;


  virtual Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) = 0;
  virtual void SetType(std::string Type) {
    this->Type=Type;
    this->ReturnType=Type;
  }
  virtual std::string GetType() {
    return Type;
  }
  virtual void SetReturnType(std::string ReturnType) {
    this->ReturnType=ReturnType;
  }

  virtual void SetIsVarLoad(bool isVarLoad) {
    this->isVarLoad=isVarLoad;
  }
  virtual bool GetIsVarLoad() {
    return isVarLoad;
  }

  virtual bool GetNameSolveToLast() {
    return NameSolveToLast;
  }
  virtual void SetNameSolveToLast(bool NameSolveToLast) {
    this->NameSolveToLast=NameSolveToLast;
  }

  virtual void SetSelf(bool Self) {
    this->isSelf=Self;
  }
  virtual bool GetSelf() {
    return isSelf;
  }

  virtual void SetSolverIncludeScope(bool SolverIncludeScope) {
    this->SolverIncludeScope=SolverIncludeScope;
  }
  virtual bool GetSolverIncludeScope() {
    return SolverIncludeScope;
  }

  virtual void SetIsAttribute(bool Attribute) {
    this->isAttribute=Attribute;
  }
  virtual bool GetIsAttribute() {
    return isAttribute;
  }
  

  virtual void SetPreDot(std::string pre_dot) {
    this->_pre_dot=pre_dot;
  }
  virtual std::string GetPreDot() {
    return _pre_dot;
  }

  virtual std::string GetName() {
    return Name;
  }
  virtual void SetName(std::string Name) {
    this->Name=Name;
  }

  
  virtual void SetIsVec(bool isVec) {
    this->isVec=isVec;
  }
  virtual bool GetIsVec() {
    return isVec;
  }

  // Tensor related
  virtual std::vector<float> GetDims() {
    return Dims;
  }
  virtual void SetDims(std::vector<float> Dims) {
    this->Dims=Dims;
  }
  virtual Value *GetTensorPtr() {
    return TensorPtr;
  }
  
};



class NameSolverAST : public ExprAST {

  public:
    std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names;
    NameSolverAST(std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names)
                    : Names(std::move(Names)) {} 
  

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


/// NumberExprAST - Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  float Val;

  public:
    NumberExprAST(float Val) : Val(Val) {
      this->SetType("float");
    } 

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};



class StringExprAST : public ExprAST {
  std::string Val;

  public:
    StringExprAST(std::string Val) : Val(Val) {
      this->SetType("str");
    } 

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};



/// VariableExprAST - Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {

  public:
    std::unique_ptr<ExprAST> NameSolver;
    VariableExprAST(std::unique_ptr<ExprAST> NameSolver, std::string Type) {
      this->isVarLoad = true;
      this->NameSolver = std::move(NameSolver);
      this->SetType(Type);
      this->NameSolver->SetType(Type);
    }

    Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
    const std::string &getName() const { return Name; }
    std::string GetName() override {
    return Name;
  }
};


class VecIdxExprAST : public ExprAST {
  
  public:
    std::unique_ptr<ExprAST> NameSolver;
    std::vector<std::unique_ptr<ExprAST>> Idx;

    VecIdxExprAST(std::unique_ptr<ExprAST> NameSolver, std::vector<std::unique_ptr<ExprAST>> Idx, std::string Type)
                  : Idx(std::move(Idx)) {
      this->isVarLoad = true;
      this->NameSolver = std::move(NameSolver);
      this->SetType(Type);
      this->NameSolver->SetType(Type);
    }

    Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
    const std::string &getName() const { return Name; }
    std::string GetName() override { return Name; }
};



class ObjectVecIdxExprAST : public ExprAST {

  public:
    std::unique_ptr<ExprAST> Vec, Idx;
    std::string _post_dot;

    ObjectVecIdxExprAST(std::unique_ptr<ExprAST> Vec, std::string _post_dot, std::unique_ptr<ExprAST> Idx)
                  : Vec(std::move(Vec)), _post_dot(_post_dot), Idx(std::move(Idx)) {
      this->isVarLoad = true;
    }

    Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};

/// VarExprAST - Expression class for var/in
class VarExprAST : public ExprAST {

  public:
    std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
    
    std::string Type;
    VarExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::string Type)
        : VarNames(std::move(VarNames)), Type(Type) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};

class StrExprAST : public ExprAST {

  public:
    std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
    StrExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames)
        : VarNames(std::move(VarNames)) {
          this->SetType("str");
        }

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


class StrVecExprAST : public ExprAST {

  public:
    std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
    std::string Type;
    
    StrVecExprAST(
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
        std::string Type)
        : VarNames(std::move(VarNames)), Type(Type) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


class NewVecExprAST : public ExprAST {

  public:
    std::vector<std::unique_ptr<ExprAST>> Values;
    std::string Type;
    
    NewVecExprAST(
        std::vector<std::unique_ptr<ExprAST>> Values,
        std::string Type)
        : Values(std::move(Values)), Type(Type) 
        {
          this->SetType(Type);
        }

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


class ObjectExprAST : public VarExprAST {

public:
  std::unique_ptr<ExprAST> Init;

  ObjectExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::unique_ptr<ExprAST> Init)
      : VarExprAST(std::move(VarNames), std::move(Type)), Init(std::move(Init)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};



class TensorExprAST : public VarExprAST {
  public:
    std::vector<std::unique_ptr<ExprAST>> V_Dims;
    std::string TensorInit;
    bool IsWeight;

    TensorExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::vector<std::unique_ptr<ExprAST>> V_Dims,
      const std::string &TensorInit, bool IsWeight)
      : VarExprAST(std::move(VarNames), std::move(Type)),
                   V_Dims(std::move(V_Dims)), TensorInit(TensorInit), IsWeight(IsWeight) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


class PinnedTensorExprAST : public VarExprAST {
  public:
    std::vector<std::unique_ptr<ExprAST>> V_Dims;
    std::string TensorInit;

    PinnedTensorExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::vector<std::unique_ptr<ExprAST>> V_Dims,
      const std::string &TensorInit)
      : VarExprAST(std::move(VarNames), std::move(Type)),
                   V_Dims(std::move(V_Dims)), TensorInit(TensorInit) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};



class Conv2dExprAST : public VarExprAST {
  public:
    std::unique_ptr<ExprAST> C, OC, Ks, Stride, Padding;
    std::string TensorInit;

    Conv2dExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::unique_ptr<ExprAST> C, std::unique_ptr<ExprAST> OC, std::unique_ptr<ExprAST> Ks,
      std::unique_ptr<ExprAST> Stride, std::unique_ptr<ExprAST> Padding,
      const std::string &TensorInit)
      : VarExprAST(std::move(VarNames), std::move(Type)),
                   C(std::move(C)), OC(std::move(OC)), Ks(std::move(Ks)),
                   Stride(std::move(Stride)), Padding(std::move(Padding)),
                   TensorInit(TensorInit) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};



class MaxPool2dExprAST : public VarExprAST {
  public:
    std::unique_ptr<ExprAST> Ks, Stride, Padding;

    MaxPool2dExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::unique_ptr<ExprAST> Ks,
      std::unique_ptr<ExprAST> Stride, std::unique_ptr<ExprAST> Padding)
      : VarExprAST(std::move(VarNames), std::move(Type)),
                   Ks(std::move(Ks)),
                   Stride(std::move(Stride)), Padding(std::move(Padding)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


class BatchNorm2dExprAST : public VarExprAST {
  public:
    std::unique_ptr<ExprAST> C;

    BatchNorm2dExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::unique_ptr<ExprAST> C)
      : VarExprAST(std::move(VarNames), std::move(Type)),
                   C(std::move(C)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


class BN2dReluExprAST : public VarExprAST {
  public:
    std::unique_ptr<ExprAST> C;

    BN2dReluExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::unique_ptr<ExprAST> C)
      : VarExprAST(std::move(VarNames), std::move(Type)),
                   C(std::move(C)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};



class LSTMExprAST : public VarExprAST {
  public:
    std::unique_ptr<ExprAST> C, OC;
    std::string TensorInit;

    LSTMExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::unique_ptr<ExprAST> C, std::unique_ptr<ExprAST> OC,
      const std::string &TensorInit)
      : VarExprAST(std::move(VarNames), std::move(Type)),
                   C(std::move(C)), OC(std::move(OC)),
                   TensorInit(TensorInit) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


class EmbeddingExprAST : public VarExprAST {
  public:
    std::unique_ptr<ExprAST> C, OC;
    std::string TensorInit;

    EmbeddingExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::unique_ptr<ExprAST> C, std::unique_ptr<ExprAST> OC,
      const std::string &TensorInit)
      : VarExprAST(std::move(VarNames), std::move(Type)),
                   C(std::move(C)), OC(std::move(OC)),
                   TensorInit(TensorInit) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};



class LinearExprAST : public VarExprAST {
  public:
    std::unique_ptr<ExprAST> C, OC;
    std::string TensorInit;
    std::vector<int> Notators;

    LinearExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::unique_ptr<ExprAST> C, std::unique_ptr<ExprAST> OC,
      std::vector<int> Notators,
      const std::string &TensorInit)
      : VarExprAST(std::move(VarNames), std::move(Type)),
                   C(std::move(C)), OC(std::move(OC)),
                   Notators(std::move(Notators)),
                   TensorInit(TensorInit) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};



class MHSAExprAST : public VarExprAST {
  public:
    std::unique_ptr<ExprAST> nh, C, T;
    std::string TensorInit;
    std::vector<int> Notators;

    MHSAExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type,
      std::unique_ptr<ExprAST> nh, std::unique_ptr<ExprAST> C, std::unique_ptr<ExprAST> T,
      std::vector<int> Notators,
      const std::string &TensorInit)
      : VarExprAST(std::move(VarNames), std::move(Type)),
                   nh(std::move(nh)), C(std::move(C)), T(std::move(T)),
                   Notators(std::move(Notators)),
                   TensorInit(TensorInit) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};



class ReluExprAST : public VarExprAST {
  public:

    ReluExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::string Type)
      : VarExprAST(std::move(VarNames), std::move(Type)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


/// UnaryExprAST - Expression class for a unary operator.
class UnaryExprAST : public ExprAST {
  char Opcode;
  std::unique_ptr<ExprAST> Operand;

public:
  UnaryExprAST(char Opcode, std::unique_ptr<ExprAST> Operand)
      : Opcode(Opcode), Operand(std::move(Operand)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


/// BinaryExprAST - Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


class BinaryTensorScalarExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryTensorScalarExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


class BinaryTensorTensorExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryTensorTensorExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


class BinaryPinnedScalarExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryPinnedScalarExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};
class BinaryPinnedAndTensorExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryPinnedAndTensorExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};

class BinaryTensorPinnedExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryTensorPinnedExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


class BinaryObjExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryObjExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


class ConcatStringsExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  ConcatStringsExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};



/// CallExprAST - Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;
  std::string Class;
  std::string PreDot;
  bool IsVarForward;
  std::string CalleeOverride;
  std::unique_ptr<ExprAST> NameSolver;

  public:
    CallExprAST(std::unique_ptr<ExprAST> NameSolver,
                const std::string &Callee,
                std::vector<std::unique_ptr<ExprAST>> Args,
                const std::string &Class,
                const std::string &PreDot,
                bool IsVarForward,
                const std::string &CalleeOverride)
        : NameSolver(std::move(NameSolver)), Callee(Callee), Args(std::move(Args)), Class(Class),
          PreDot(PreDot), IsVarForward(IsVarForward), CalleeOverride(CalleeOverride) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};

class ReturnExprAST : public ExprAST {

  public:
    std::vector<std::unique_ptr<ExprAST>> Vars;
    std::vector<bool> IsAs;
    std::vector<std::unique_ptr<ExprAST>> Destiny;
    
    ReturnExprAST(std::vector<std::unique_ptr<ExprAST>> Vars, std::vector<bool> IsAs,
                  std::vector<std::unique_ptr<ExprAST>> Destiny)
        : Vars(std::move(Vars)), IsAs(std::move(IsAs)), Destiny(std::move(Destiny)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


/// IfExprAST - Expression class for if/then/else.
class IfExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Cond;
  std::vector<std::unique_ptr<ExprAST>> Then, Else;

  public:
    IfExprAST(std::unique_ptr<ExprAST> Cond,
              std::vector<std::unique_ptr<ExprAST>> Then,
              std::vector<std::unique_ptr<ExprAST>> Else)
        : Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};

/// ForExprAST - Expression class for for.
class ForExprAST : public ExprAST {
  std::string VarName;
  std::unique_ptr<ExprAST> Start, End, Step;
  std::vector<std::unique_ptr<ExprAST>> Body;

  public:
    ForExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Start,
              std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
              std::vector<std::unique_ptr<ExprAST>> Body)
        : VarName(VarName), Start(std::move(Start)), End(std::move(End)),
          Step(std::move(Step)), Body(std::move(Body)) {}

  Value *codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};

/// WhileExprAST - Expression class for while.
class WhileExprAST : public ExprAST {
	std::unique_ptr<ExprAST> Cond;
  std::vector<std::unique_ptr<ExprAST>> Body;

  public:
    WhileExprAST(std::unique_ptr<ExprAST> Cond, std::vector<std::unique_ptr<ExprAST>> Body)
      : Cond(std::move(Cond)), Body(std::move(Body)) {}

	Value* codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


/// AsyncExprAST - Expression class for async.
class AsyncExprAST : public ExprAST {
	std::vector<std::unique_ptr<ExprAST>> Body;

  public:
    AsyncExprAST(std::vector<std::unique_ptr<ExprAST>> Body)
      : Body(std::move(Body)) {}

	Value* codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


/// FinishExprAST - Expression class for finish/async.
class FinishExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> Bodies;
  std::vector<bool> IsAsync;

  public:
    FinishExprAST(std::vector<std::unique_ptr<ExprAST>> Bodies,
                  std::vector<bool> IsAsync)
            : Bodies(std::move(Bodies)), IsAsync(std::move(IsAsync)) {}


	Value* codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};


/// LockExprAST
class LockExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> Bodies;
  std::string Name;

  public:
    LockExprAST(std::vector<std::unique_ptr<ExprAST>> Bodies,
                std::string Name)
            : Bodies(std::move(Bodies)), Name(Name) {}


	Value* codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};
/// NoGradExprAST
class NoGradExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> Bodies;

  public:
    NoGradExprAST(std::vector<std::unique_ptr<ExprAST>> Bodies)
            : Bodies(std::move(Bodies)) {}


	Value* codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) override;
};





/// PrototypeAST - This class represents the "prototype" for a function,
/// which captures its name, and its argument names (thus implicitly the number
/// of arguments the function takes), as well as if it is an operator.
class PrototypeAST {

  std::string Name;
  std::string Class;
  std::string Method;

  std::vector<std::string> Args;
  std::vector<std::string> Types;
  bool IsOperator;
  unsigned Precedence; // Precedence if a binary op.

  public:
    PrototypeAST(const std::string &Name, const std::string &Class, const std::string &Method,
                std::vector<std::string> Args,
                std::vector<std::string> Types,
                bool IsOperator = false, unsigned Prec = 0)
        : Name(Name), Class(Class), Method(Method), Args(std::move(Args)), Types(std::move(Types)),
          IsOperator(IsOperator), Precedence(Prec) {}

  Function *codegen();
  const std::string &getName() const { return Name; }
  const std::string &getClass() const { return Class; }
  const std::string &getMethod() const { return Method; }

  bool isUnaryOp() const { return IsOperator && Args.size() == 1; }
  bool isBinaryOp() const { return IsOperator && Args.size() == 2; }

  char getOperatorName() const {
    assert(isUnaryOp() || isBinaryOp());
    return Name[Name.size() - 1];
  }



  unsigned getBinaryPrecedence() const { return Precedence; }
};



//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//
//global

/// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
/// token the parser is looking at.  getNextToken reads another token from the
/// lexer and updates CurTok with its results.
static int CurTok;
static int getNextToken() { return CurTok = get_token(); }

/// BinopPrecedence - This holds the precedence for each binary operator that is
/// defined.
static std::map<char, int> BinopPrecedence;

/// get_tokenPrecedence - Get the precedence of the pending binary operator token.
static int get_tokenPrecedence() {
  if (CurTok==tok_space)
    return 1;


  if (BinopPrecedence.find(CurTok) == BinopPrecedence.end()) // if not found
    return -1;

  // Make sure it's a declared binop.
  int TokPrec = BinopPrecedence[CurTok];
  if (TokPrec <= 0)
    return -1;
  return TokPrec;
}



/// LogError* - These are little helper functions for error handling.
std::unique_ptr<ExprAST> LogErrorS(std::string Str) {
  ShallCodegen = false;
  //fprintf(stderr, "\033[31m Error: \033[0m%s\n", Str);
  if (Str!=" ")
    std::cout << "\nLine: " << LineCounter << "\n   \033[31m Error: \033[0m " << Str << "\n\n";
  
  
  return nullptr;
}

std::unique_ptr<ExprAST> LogError(std::string Str) {
  //fprintf(stderr, "\033[31m Error: \033[0m%s\n", Str);
  LogErrorS(Str);

  while(CurTok!=tok_space && CurTok!=',' && CurTok!=')' && !in_char(CurTok, terminal_tokens))
    getNextToken();
  
  return nullptr;
}


std::unique_ptr<ExprAST> LogError_toNextToken(std::string Str) {
  //fprintf(stderr, "\033[31m Error: \033[0m%s\n", Str);
  LogErrorS(Str);

  getNextToken();
  
  return nullptr;
}


std::unique_ptr<ExprAST> LogErrorBreakLine(std::string Str) {
  //fprintf(stderr, "\033[31m Error: \033[0m%s\n", Str);
  LogErrorS(Str);

  while(CurTok!=tok_space && !in_char(CurTok, terminal_tokens))
    getNextToken();

  if (CurTok==tok_space)
    getNextToken();
  
  return nullptr;
}

void LogWarning(const char *Str) {
  std::cout << "\nLine: " << LineCounter << "\n   \033[33m Aviso: \033[0m " << Str << "\n\n";
}

// Modified LogError function with token parameter
std::unique_ptr<ExprAST> LogErrorT(int CurTok) {
  ShallCodegen = false;
  //char buf[100];
  //snprintf(buf, sizeof(buf), "token %d inesperado.", CurTok);
  //fprintf(stderr, "\033[31mError: \033[0m%s\n", buf);
  std::cout << "\nLine: " << LineCounter << "\n   \033[31m Error: \033[0mUnexpected token " << ReverseToken(CurTok) << ". Expected an expression.\n\n";
  
  while(CurTok!=tok_space && !in_char(CurTok, terminal_tokens))
    getNextToken();

  return nullptr;
}


std::unique_ptr<PrototypeAST> LogErrorP(const char *Str) {
  LogError(Str);
  while(CurTok!=tok_space && !in_char(CurTok, terminal_tokens))
    getNextToken();
  return nullptr;
}


std::unique_ptr<PrototypeAST> LogErrorP_to_comma(const char *Str) {
  LogError(Str);
  while(CurTok!=tok_space && CurTok!=',' && CurTok!=')' && !in_char(CurTok, terminal_tokens))
  {
    std::cout << "LogErrorP: " << IdentifierStr << "\n";
    
    getNextToken();
    }
  return nullptr;
}

Value *LogErrorV(std::string Str) {
  LogError(Str);
  return nullptr;
}

static std::unique_ptr<ExprAST> ParseExpression(std::string class_name="");
static std::unique_ptr<ExprAST> ParsePrimary(std::string class_name);

/// numberexpr ::= number
static std::unique_ptr<ExprAST> ParseNumberExpr() {
  auto Result = std::make_unique<NumberExprAST>(NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

static std::unique_ptr<ExprAST> ParseStringExpr() {
  auto Result = std::make_unique<StringExprAST>(IdentifierStr);
  getNextToken(); // consume the "
  return std::move(Result);
}



/// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> ParseParenExpr(std::string class_name="") {
  getNextToken(); // eat (.
  auto V = ParseExpression(class_name);
  if (!V)
    return nullptr;

  if (CurTok != ')')
    return LogError("Expected ')' on parenthesis expression.");
  

  getNextToken(); // eat ).
  return V;
}

//global
std::vector<std::string> tensorVars;
std::vector<std::string> pinnedTensorVars;
std::vector<std::string> floatVars;
std::vector<std::string> strVars;
std::vector<std::string> objectVars;
std::vector<std::string> globalVars;
std::map<std::string, std::string> functionVars;
std::map<std::string, std::string> floatFunctions;
std::map<std::string, std::string> stringMethods;
std::vector<std::string> str_vecVars;
std::vector<std::string> float_vecVars;
std::map<std::string, pthread_mutex_t *> lockVars;



//global
static std::vector<std::string> Classes;
static std::map<std::string, std::string> Object_toClass;
static std::map<std::string, std::string> Object_toClassVec;


static std::unique_ptr<ExprAST> ParseObjectInstantiationExpr(std::string _class, std::string class_name) {
  getNextToken();
  //std::cout << "Object name: " << IdentifierStr << " and Class: " << Classes[i]<< "\n";
  bool is_vec=false;
  bool is_self=false;
  bool is_attr=false;
  std::string pre_dot="";
  std::unique_ptr<ExprAST> VecInitSize = nullptr;
      
  //std::cout << "\n\n\n\nCUR TOK IS: " << ReverseToken(CurTok) << "\n\n\n\n\n\n";
  if (CurTok==tok_vec)
  {
    getNextToken();
    is_vec=true;
        
    if(CurTok=='[')
    {
      getNextToken();
      VecInitSize = ParsePrimary(class_name);
      if (CurTok!=']')
        LogError("Expected ] at object vec");
      getNextToken();
    }
    Object_toClassVec[IdentifierStr] = _class; //todo: this doesn't deal with self. expr
  }


  if (CurTok==tok_self)
  {
    getNextToken();
    is_self=true;
  }


  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  while (true) {
    std::string Name = IdentifierStr;
    objectVars.push_back(Name);
    if (!is_vec)
      Object_toClass[IdentifierStr] = _class;
    getNextToken(); // eat identifier.

        
    std::unique_ptr<ExprAST> Init = nullptr;  
    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected object identifier names.");
  }

  auto aux = std::make_unique<ObjectExprAST>(std::move(VarNames), "object", std::move(VecInitSize));
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);
  aux->SetIsVec(is_vec);

  if (CurTok==tok_space)
    getNextToken();

  return aux;
}


static std::vector<std::unique_ptr<ExprAST>> ParseIdx(std::string class_name="") {

  std::vector<std::unique_ptr<ExprAST>> Idx;
    
  Idx.push_back(ParseExpression(class_name));
  while(CurTok==',')
  {
    getNextToken(); // eat ,
    Idx.push_back(ParseExpression(class_name));
  }
  Idx.push_back(std::make_unique<NumberExprAST>(TERMINATE_VARARG));

  return Idx;
}



/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
static std::unique_ptr<ExprAST> ParseIdentifierExpr(std::string class_name="") {
  
  for(int i=0; i<Classes.size(); i++)
    if(IdentifierStr==Classes[i])  // Object object
      return ParseObjectInstantiationExpr(Classes[i], class_name);

  std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names;
  std::string IdName, type;
  IdName = IdentifierStr;
  //std::cout << "Identifier " << IdName <<  "\n";
  getNextToken(); // eat identifier.
  
  Names.push_back(std::make_tuple(IdName, type_var, std::vector<std::unique_ptr<ExprAST>>{}));
  if (CurTok != '(' && CurTok != '[') // Simple variable ref.
  {
    if (in_str(IdName, pinnedTensorVars))
      type = "pinned_tensor";
    if (in_str(IdName, tensorVars))
      type = "tensor";
    if (in_str(IdName, floatVars))
      type = "float";
    if (in_str(IdName, strVars))
      type = "str";
    if (in_str(IdName, objectVars))
      type = "object";


    auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
    auto aux = std::make_unique<VariableExprAST>(std::move(name_solver_expr), type);
    
    
    if (CurTok==tok_space)
      getNextToken();

    return aux;
  }



  if (CurTok=='[')
  {
    getNextToken(); // eat [
    
    std::vector<std::unique_ptr<ExprAST>> Idx;
    Idx = ParseIdx(class_name);
    
    if (in_str(IdName, str_vecVars))
      type = "str_vec";
    if (in_str(IdName, float_vecVars))
      type = "float_vec";
    if (in_str(IdName, tensorVars))
      type = "tensor";

    auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
    auto aux = std::make_unique<VecIdxExprAST>(std::move(name_solver_expr), std::move(Idx), type);
    aux->SetIsVec(true);
    
    getNextToken(); // eat ]
    
    return std::move(aux);
  }
  

  // Call.
  getNextToken(); // eat (
  
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (true) {
      
      if (auto Arg = ParseExpression(class_name))
        Args.push_back(std::move(Arg));
      else
        return nullptr;
      

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        return LogError("Esperado ')' ou ',' na lista de argumentos");
      getNextToken();
    }
  }

  // varargs
  if (in_str(IdName, vararg_methods))
    Args.push_back(std::make_unique<NumberExprAST>(TERMINATE_VARARG));
  
  

  // Eat the ')'.
  getNextToken();

  bool is_var_forward = false;
  bool return_tensor = false;
  std::string callee_override = "none";
  if (functionVars.find(IdName) != functionVars.end()) // if found
  {
    is_var_forward = true;
    return_tensor = true;
    callee_override = functionVars[IdName];
  }
  if (floatFunctions.find(IdName) != floatFunctions.end()) // if found
  {
    is_var_forward = true;
    callee_override = floatFunctions[IdName];
  }
  if (IdName=="to_float")
  {
    callee_override = "ToFloat";
    is_var_forward = true;
  }
  
  auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
  name_solver_expr->SetNameSolveToLast(false);
  auto aux = std::make_unique<CallExprAST>(std::move(name_solver_expr), IdName, std::move(Args),
                                                "None", "None", is_var_forward, callee_override);

  
  
  if (in_str(IdName, return_tensor_fn) || return_tensor)
    aux->SetType("tensor");
  if (in_str(IdName, return_pinned_methods))
    aux->SetType("pinned_tensor");
  if (in_str(IdName, return_string_fn))
    aux->SetType("str");

  return aux;
}





/// ifexpr ::= 'if' expression 'then' expression 'else' expression
static std::unique_ptr<ExprAST> ParseIfExpr(std::string class_name="") {
  
  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the if.

  
  //std::cout << "If tabs level: " << cur_level_tabs <<  "\n";
  

  // condition.
  auto Cond = ParseExpression(class_name);
  if (!Cond)
    return nullptr;

  if(CurTok==tok_space)
    getNextToken();

  
  std::vector<std::unique_ptr<ExprAST>> Then, Else;
  
  while(true)
  {
    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
      break;

    while (CurTok == tok_space)
      getNextToken();
    
    if (SeenTabs <= cur_level_tabs)
      break;
    
    auto body = ParseExpression(class_name);
    if (!body)
      return nullptr;
    Then.push_back(std::move(body));
    
  }
  
  if (Then.size()==0)
  {
    LogError("Then is null");
    return nullptr;
  }
  
  
  if(CurTok == tok_space)
    getNextToken();

  //std::cout << "\n\nIf else token: " << ReverseToken(CurTok) <<  "\n\n\n";

  if (CurTok != tok_else) {
    Else.push_back(std::make_unique<NumberExprAST>(0));

    return std::make_unique<IfExprAST>(std::move(Cond), std::move(Then),
                                      std::move(Else));
  }
  else {
    getNextToken(); //eat else
    if(CurTok != tok_space)
      LogError("else requer barra de espaço.");
    getNextToken();

    while(true)
    {

      if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
        break;

      while (CurTok == tok_space)
        getNextToken();

      if (SeenTabs <= cur_level_tabs)
        break;
      
      auto body = ParseExpression(class_name);
      if (!body)
        return nullptr;
      Else.push_back(std::move(body));
    }

  
    if (CurTok==tok_space)
      getNextToken();

    return std::make_unique<IfExprAST>(std::move(Cond), std::move(Then),
                                      std::move(Else));
  }
}




static std::vector<std::unique_ptr<ExprAST>> ParseIdentedBodies(int cur_level_tabs, std::string class_name="")
{
  std::vector<std::unique_ptr<ExprAST>> Body, NullBody;
  //std::cout << "\nSeen tabs on for body POST: " << SeenTabs << "\n\n";
  if (CurTok==tok_space)
    getNextToken();

  while(true)
  {
    //std::cout << "\n\nParsing new expression with tabs: " << SeenTabs << " tok: " << ReverseToken(CurTok) << "\n";
    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
    {
      //std::cout << "Breaking for with cur tok: " << ReverseToken(CurTok) << " Seen Tabs:" << SeenTabs <<  "\n";
      break;
    } 
    //std::cout << "\nSeen tabs on for body: " << SeenTabs << "\nCur tok: " << ReverseToken(CurTok) << "\n\n";

    while (CurTok == tok_space)
    {
      //std::cout << "\nJumping tok space\n\n";
      getNextToken();
    }

    //std::cout << "Post space has " << SeenTabs << " tabs.\n";
    if (SeenTabs <= cur_level_tabs)
      break;


    auto body = ParseExpression(class_name);
    if (!body)
      return std::move(NullBody);
    Body.push_back(std::move(body));
    //getNextToken();
  }

  if (CurTok==tok_space)
    getNextToken();

  return std::move(Body);
}


/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
static std::unique_ptr<ExprAST> ParseForExpr(std::string class_name="") {

  int cur_level_tabs = SeenTabs;

  //std::cout << "\nSeen tabs on for: " << SeenTabs << "\n\n";

  getNextToken(); // eat the for.


  if (CurTok != tok_identifier)
    return LogError("Expected for's control variable identifier.");

  std::string IdName = IdentifierStr;
  getNextToken(); // eat identifier.

  floatVars.push_back(IdName);

  if (CurTok != '=')
    return LogError("Expected for's control variable initial value.");
  getNextToken(); // eat '='.

  auto Start = ParseExpression(class_name);
  if (!Start)
    return nullptr;
  if (CurTok != ',')
    return LogError("Expected ',' after for's control variable initial value.");
  getNextToken();



  auto End = ParseExpression(class_name);
  if (!End)
    return nullptr;

  


  std::unique_ptr<ExprAST> Step = std::make_unique<NumberExprAST>(1.0);
  if (CurTok == ',') { // The step value is optional.
    getNextToken();
    auto aux = ParseExpression(class_name);
    if (aux)
      Step = std::move(aux);
  }
  
  std::vector<std::unique_ptr<ExprAST>> Body;

  Body = ParseIdentedBodies(cur_level_tabs, class_name);

  return std::make_unique<ForExprAST>(IdName, std::move(Start), std::move(End),
                                       std::move(Step), std::move(Body));
}



/// whileexpr ::= 'while' identifier '=' expr ',' expr (',' expr)? 'in' expression
static std::unique_ptr<ExprAST> ParseWhileExpr(std::string class_name="") {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the while.


  //if (CurTok != tok_identifier)
  //  return LogError("Identificador da variável de controle esperado depois do while.");


  auto Cond = ParseExpression(class_name);
  if (!Cond)
    return nullptr;
  
  std::vector<std::unique_ptr<ExprAST>> Body = ParseIdentedBodies(cur_level_tabs, class_name);

  return std::make_unique<WhileExprAST>(std::move(Cond), std::move(Body));
}



static std::unique_ptr<ExprAST> ParseAsyncExpr(std::string class_name="") {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat the async.

  
  std::vector<std::unique_ptr<ExprAST>> Bodies;
  
  //std::cout << "Pre expression token: " << ReverseToken(CurTok) << "\n";

  if (CurTok != tok_space)
    Bodies.push_back(std::move(ParseExpression(class_name)));
  else
    Bodies = ParseIdentedBodies(cur_level_tabs, class_name);
  
  
  
  //std::cout << "Post async: " << ReverseToken(CurTok) << "\n";

  return std::make_unique<AsyncExprAST>(std::move(Bodies));
}



static std::unique_ptr<ExprAST> ParseFinishExpr(std::string class_name="") {

  int cur_level_tabs = SeenTabs;
  //std::cout << "Finish tabs level: " << cur_level_tabs <<  "\n";

  getNextToken(); // eat the finish.


  std::vector<std::unique_ptr<ExprAST>> Bodies;
  std::vector<bool> IsAsync;
  

  if (CurTok!=tok_space)
    LogError("Finish requires line break.");
  getNextToken(); 


  while(!in_char(CurTok, terminal_tokens))
  {

    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
    {
      //std::cout << "Breaking finish with cur tok: " << ReverseToken(CurTok) << "\n";
      //std::cout << "Current tabs: " << SeenTabs << "\n";
      break;
    }
    //std::cout << "\nSeen tabs on finish body: " << SeenTabs << "\nCur tok: " << CurTok << "\n\n";




    if (CurTok==tok_space)
      getNextToken();
    

    /*
    if (CurTok == tok_async)
    {
      Bodies.push_back(std::move(ParseAsyncExpr(class_name)));
      IsAsync.push_back(true);
    }
    else
    {
      Bodies.push_back(std::move(ParseExpression(class_name)));
      IsAsync.push_back(false);
    }
    */
    Bodies.push_back(std::move(ParseExpression(class_name)));
    IsAsync.push_back(false);
  }


  return std::make_unique<FinishExprAST>(std::move(Bodies),
                                         std::move(IsAsync));
}


/// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
static std::unique_ptr<ExprAST> ParseVarExpr(std::string class_name="") {
  getNextToken(); // eat the var.
  

  // mem2reg is alloca-driven: it looks for allocas and if it can handle them, it promotes them. It DOES NOT APPLY TO GLOBAL variables or heap allocations.
  // mem2reg only promotes allocas whose uses are direct loads and stores. If the address of the stack object is passed to a function,
  //or if any funny pointer arithmetic is involved, the alloca will not be promoted.

  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("Expected an identifier after float.");

  while (true) {
    std::string Name = IdentifierStr;
    floatVars.push_back(IdentifierStr);
    getNextToken(); // eat identifier.

    // Read the optional initializer.
    std::unique_ptr<ExprAST> Init = nullptr;
    if (CurTok == '=')
    {
      getNextToken(); // eat the '='.

      Init = ParseExpression(class_name);
      if (!Init)
        return nullptr;
    }

    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers after float.");
  }

  if (CurTok==tok_space)
    getNextToken();


  return std::make_unique<VarExprAST>(std::move(VarNames), "var");
}




static std::unique_ptr<ExprAST> ParseStrExpr() {
  getNextToken(); // eat str
  

  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("Esperado identificador após var.");

  
  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true; //TODO: set self per VarName instead.
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    getNextToken();
  }

  while (true) {
    std::string Name = IdentifierStr;
    strVars.push_back(Name);
    getNextToken(); // eat identifier.

    // Read the optional initializer.
    std::unique_ptr<ExprAST> Init = nullptr;
    if (CurTok == '=') {
      getNextToken(); // eat the '='.

      Init = ParseStringExpr();
      if (!Init)
        return nullptr;
    }

    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Esperado um ou mais identificadores após var.");
  }

  auto aux = std::make_unique<StrExprAST>(std::move(VarNames));

  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);
  

  if (CurTok==tok_space)
    getNextToken();


  return aux;
}

static std::unique_ptr<ExprAST> ParseNewVector(std::string class_name="") {
  std::cout << "Parsing new vector" << ReverseToken(CurTok)  << "\n";
  getNextToken(); // [
  std::vector<std::unique_ptr<ExprAST>> values = ParseIdx(class_name);
  getNextToken(); // ]


  if (CurTok==tok_space)
    getNextToken();
  
  values.push_back(std::make_unique<NumberExprAST>(TERMINATE_VARARG));

  //TODO: vector for other types
  return std::make_unique<NewVecExprAST>(std::move(values), "tensor");
}


static std::unique_ptr<ExprAST> ParseStrVecExpr() {
  int vec_type = CurTok;
  std::string vec_type_str;

  if (vec_type==tok_str_vec)
    vec_type_str = "str";
  if (vec_type==tok_float_vec)
    vec_type_str = "float";
    
  getNextToken(); // eat str_vec

  
  
  
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("Expected identifier after vector var.");

  

  while (true) {
    std::string Name = IdentifierStr;
    getNextToken(); // eat identifier.

    // Read the optional initializer.
    std::unique_ptr<ExprAST> Init = nullptr;
    if (CurTok == '=') {
      getNextToken(); // eat the '='.

      Init = ParseStringExpr();
      if (!Init)
        return nullptr;
    }

    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    if (vec_type==tok_str_vec)
      str_vecVars.push_back(Name);
    if (vec_type==tok_float_vec)
      float_vecVars.push_back(Name);
    

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Esperado um ou mais identificadores após var.");
  }

  if (CurTok==tok_space)
    getNextToken();

  return std::make_unique<StrVecExprAST>(std::move(VarNames), vec_type_str);
}


struct int_vec{
  int *vec;
  int size;
};

int_vec *CreateIntVec(int *vec, int size)
{

  int_vec *ivec = new int_vec();
  ivec->vec = vec;
  ivec->size = size;
  return ivec;
}

int_vec *SetNotators(std::vector<int> Notators)
{

  int_vec *ivec = new int_vec();

  int notators_size = Notators.size();
  int *notators = new int[notators_size];
  for (int i=0; i<notators_size; ++i)
    notators[i] = Notators[i];

  ivec->vec = notators;
  ivec->size = notators_size;
  return ivec;
}


float FirstNonzero(int *vec, int size)
{  
  /*
  std::cout << "[";
  for (int i=0; i<size; i++)
    std::cout << vec[i] << ", ";
  std::cout << "]" << "\n";
  */

  float idx = -1;
  for (int i=0; i<size; i++)
    if (vec[i]!=0)
    {    
      idx = i;
      break;
    }
  return idx;
}



// Cuda Parallellism

struct CudaStreams {
  cudaStream_t stream;
  int idx;
};
constexpr int num_parallel_streams = 32;
CudaStreams *parallel_streams[num_parallel_streams];
cudaEvent_t parallel_events[num_parallel_streams];

std::vector<cudaEvent_t> Registered_Events;

int open_streams[num_parallel_streams];

CudaStreams *AllocateStream()
{
  int free_stream = FirstNonzero(open_streams, num_parallel_streams);
  if (free_stream<0)
    LogErrorS("Failed to allocate a cuda stream. Probably loading too many different tensors.");
  open_streams[free_stream] = 0;
  //std::cout << "Allocating stream " << free_stream << "\n";
  return parallel_streams[free_stream];
}
void SynchronizeStream(CudaStreams *cuda_stream)
{
  //std::cout << "Synchronizing stream " << cuda_stream->idx << "\n";
  cudaStreamSynchronize(cuda_stream);
  open_streams[cuda_stream->idx] = 1;
}

CudaStreams *main_stream, *backward_stream;
static std::map<int, cudaStream_t> ThreadsStream;


void RegisterEvent(cudaStream_t stream)
{
  //TODO: does this work inside threads?

  cudaEvent_t event;
  cudaEventCreate(&event);

  cudaEventRecord(event, stream);

  Registered_Events.push_back(event);
}
void WaitForAllEvents()
{
  while(Registered_Events.size()>0)
  {
    cudaEvent_t event = Registered_Events.back();
    Registered_Events.pop_back();

    cudaStreamWaitEvent(main_stream, event, 0);
    cudaEventDestroy(event);
  }
}


void StreamAwaitStreamB(cudaStream_t A, cudaStream_t B)
{
  // Create an event
  cudaEvent_t event;
  cudaEventCreate(&event);

  // Record the event when the kernel finishes execution on 'stream'
  cudaEventRecord(event, B);

  cudaStreamWaitEvent(A, event, 0);
  cudaEventDestroy(event);
}


int ASYNC_LOADER_THREADS = 6;

void copyChunk(float* d_data, const float* h_data, int offset, float size, cudaStream_t stream) {
    cudaMemcpyAsync(d_data + offset, h_data + offset, size*sizeof(float), cudaMemcpyHostToDevice, stream);
}

struct Loader {
    std::vector<std::thread> threads;
    std::vector<CudaStreams *> streams;

    void Load(float *tensor_ptr, const float *tensor_cpu, int all_dims_prod) {

      float quotient = std::floor(all_dims_prod / ASYNC_LOADER_THREADS);
      float remainder = all_dims_prod % ASYNC_LOADER_THREADS;


      std::vector<float> dims_prods;

      for(int i=0; i<ASYNC_LOADER_THREADS-1; i++)
        dims_prods.push_back(quotient);
      dims_prods.push_back(quotient+remainder);


      float offset, size;
      offset = 0;
      for(int i=0; i<ASYNC_LOADER_THREADS; i++)
      {
        size = dims_prods[i];
        CudaStreams *cuda_stream = AllocateStream();

        //copyChunk(tensor_ptr, tensor_cpu, offset, size, cuda_stream);
        //threads.push_back(std::thread(copyChunk, tensor_ptr, tensor_cpu, offset, size, cuda_stream));

        cudaMemcpyAsync(tensor_ptr + (int)offset, tensor_cpu + (int)offset, size*sizeof(float), cudaMemcpyHostToDevice, cuda_stream);

        streams.push_back(cuda_stream);
        offset += size;
      }
    }
    
    void Sync()
    {
      for(int i=0; i<ASYNC_LOADER_THREADS; i++)
      {
        SynchronizeStream(streams[i]);
        //threads[i].join();
      }
      streams.clear();
      //threads.clear();
    }
};



struct Tensor {
  float *tensor_ptr;
  float *cpu_tensor_ptr;
  std::vector<float> dims;
  float dims_prod;
  float *b=nullptr;
  float *dy=nullptr;
  float scalar;
  int b_size=0;
  int thread_id;
  bool is_pinned;

  CudaStreams *cuda_stream = nullptr;
  Loader *loader = nullptr;

  bool leaf, weight, from_grad_or_load;
  std::string view_of = "";
  std::string name;
  std::string scopeless_name;
  std::string from_cudnn;
  int op;

  Tensor *R_Node, *L_Node, *Sparse_Idx_Tensor;
  bool visited;

  void NewNullTensor()
  {
    tensor_ptr = nullptr;
    dims = {0};
    dims_prod = 0;
    cpu_tensor_ptr = nullptr;
    L_Node=nullptr;
    R_Node=nullptr;
    dy=nullptr;
    visited=false;
    weight=false;
    from_grad_or_load=false;
    cuda_stream = nullptr;
    loader = nullptr;
    from_cudnn = "";
    is_pinned=false;
    thread_id = 0;
    scalar=1;
    Sparse_Idx_Tensor=nullptr;
  }

  void NewTensor(float *new_tensor_ptr, std::vector<float> new_dims, float new_dims_prod,
                 bool new_is_leaf, std::string new_name, CudaStreams *_cuda_stream=nullptr, Loader *_loader=nullptr){
    tensor_ptr = new_tensor_ptr;
    dims = new_dims;
    dims_prod = new_dims_prod;
    leaf = new_is_leaf;
    name = new_name;
    cpu_tensor_ptr = nullptr;
    op=leaf;
    L_Node=nullptr;
    R_Node=nullptr;
    dy=nullptr;
    visited=false;
    weight=false;
    op=tensor_leaf;
    from_grad_or_load=false;
    cuda_stream = _cuda_stream;
    loader = _loader;
    from_cudnn = "";
    is_pinned=false;
    thread_id = 0;
    scalar=1;
    Sparse_Idx_Tensor=nullptr;
  }

  void NewPinned(float *new_tensor_ptr, float *new_cpu_tensor_ptr,
                 std::vector<float> new_dims, float new_dims_prod,
                 bool new_is_leaf, std::string new_name){
    tensor_ptr = new_tensor_ptr;
    cpu_tensor_ptr = new_cpu_tensor_ptr;
    dims = new_dims;
    dims_prod = new_dims_prod;
    leaf = new_is_leaf;
    name = new_name;
    weight=false;
    from_grad_or_load=true;
    is_pinned=true;
    thread_id = 0;
    Sparse_Idx_Tensor=nullptr;
  }

  void AttrTensor(float *new_tensor_ptr, std::vector<float> new_dims, float new_dims_prod, CudaStreams *_cuda_stream=nullptr, Loader *_loader=nullptr){
    tensor_ptr = new_tensor_ptr;
    dims = new_dims;
    dims_prod = new_dims_prod;
    cuda_stream = _cuda_stream;
    loader = _loader;
    is_pinned=false;
  }

  
  void AttrNodes(Tensor *new_L_Tensor, Tensor *new_R_Tensor, int op_type)
  {
    L_Node = new_L_Tensor;
    R_Node = new_R_Tensor;
    op = op_type;
    leaf=false;
    visited=false;
    dy=nullptr;
    weight=false;
    from_grad_or_load = ((from_grad_or_load||new_L_Tensor->from_grad_or_load||new_R_Tensor->from_grad_or_load)&&!in_int(op, gradless_ops));
    is_pinned=false;
  }

  void AttrLNode(Tensor *new_L_Tensor, int op_type)
  {
    L_Node = new_L_Tensor;
    R_Node=nullptr;
    op = op_type;
    leaf=false;
    visited=false;
    dy=nullptr;
    weight=false;
    from_grad_or_load = ((from_grad_or_load||new_L_Tensor->from_grad_or_load)&&!in_int(op, gradless_ops));
    is_pinned=false;
  }

  void AttributionBackwardNode(std::string _name, Tensor *new_R_Tensor)
  {
    name = _name;
    R_Node = new_R_Tensor;
    op = attribution;
    leaf=false;
    visited=false;
    
    L_Node=nullptr;
    dy=nullptr;
    weight=false;
    is_pinned=false;
  }
  void SetIsWeight()
  {
    weight=true;
    from_grad_or_load=true;
    is_pinned=false;
  }
  void SetBias(float *b, int b_size)
  {
    this->b=b;
    this->b_size=b_size;
    leaf=true;
    is_pinned=false;
  }
  void Sync()
  {
    if(loader!=nullptr)
    {
      loader->Sync();
      delete loader;
      loader=nullptr;
    }
    if(cuda_stream!=nullptr)
    {
      SynchronizeStream(cuda_stream);
      cuda_stream=nullptr;
    }
  }
};

Tensor *createTensor(float* tensor_ptr, const std::vector<float>& dims, float kDataLen,
                     bool is_leaf, std::string name, CudaStreams *_cuda_stream=nullptr, Loader *_loader=nullptr) {
    Tensor *new_tensor = new Tensor();
    new_tensor->NewTensor(tensor_ptr, dims, kDataLen, is_leaf, name, _cuda_stream, _loader);
    return new_tensor;
}
Tensor *createPinned(float* tensor_ptr, float *tensor_cpu, const std::vector<float>& dims, float kDataLen,
                     std::string name) {
    Tensor *new_tensor = new Tensor();
    new_tensor->NewPinned(tensor_ptr, tensor_cpu, dims, kDataLen, true, name);
    return new_tensor;
}
Tensor *createBackward(std::string name, Tensor *tensor) {
    Tensor *new_tensor = new Tensor();
    new_tensor->AttributionBackwardNode(name, tensor);
    return new_tensor;
}
Tensor *wrapTensorWithDetached(Tensor* tensor) {
    /*
    Tensor *new_tensor = new Tensor();

    new_tensor->NewNullTensor();
    new_tensor->AttrLNode(tensor, detach_op);
    new_tensor->tensor_ptr = tensor->tensor_ptr;
    new_tensor->dims_prod = tensor->dims_prod;
    new_tensor->dims = tensor->dims;
    
    return new_tensor;
    */
    
    tensor->op = detach_op;
    return tensor;
}


bool in_tensor_ptr_vec(Tensor *value, const std::vector<Tensor *>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}




// Cleaners
std::map<std::string, float *> var_to_grad;
std::vector<std::tuple<float, float *, std::string>> backprop_tensors_to_pool;
std::vector<float *> tensors_sent_to_pool;
std::vector<Tensor *> backprop_Tensors_to_free;
std::vector<Tensor *> backprop_Tensors_to_save;

void to_free_tensor(Tensor *tensor_ptr)
{
  if(!in_tensor_ptr_vec(tensor_ptr, backprop_Tensors_to_free))
    backprop_Tensors_to_free.push_back(tensor_ptr);
}
void to_pool(float dims_prod, float *tensor_ptr, std::string from)
{
  if (!in_float_ptr_vec(tensor_ptr, tensors_sent_to_pool))
  {
    backprop_tensors_to_pool.push_back(std::make_tuple(dims_prod, tensor_ptr, from));
    tensors_sent_to_pool.push_back(tensor_ptr);
  }
}
void save_from_pool(Tensor *tensor_ptr)
{
  if(!in_tensor_ptr_vec(tensor_ptr, backprop_Tensors_to_save))
    backprop_Tensors_to_save.push_back(tensor_ptr);
}


std::map<std::string, std::vector<std::tuple<float, float*,std::string>>> forward_tensors_to_pool;
std::map<std::string, std::vector<float*>> forward_tensors_sent_to_pool;
std::map<std::string, std::vector<Tensor*>> forward_Tensors_to_free;
std::map<std::string, std::map<std::string, float*>> scope_tensors; // records last version of a tensor //todo: is this one actually used?

void to_free_tensor_forward(Tensor *tensor_ptr, std::string scope)
{
  if(!in_tensor_ptr_vec(tensor_ptr, forward_Tensors_to_free[scope]))
    forward_Tensors_to_free[scope].push_back(tensor_ptr);
}
void to_pool_forward(float dims_prod, float *tensor_ptr, std::string scope, std::string from)
{
  if (!in_float_ptr_vec(tensor_ptr, forward_tensors_sent_to_pool[scope]))
  {
    forward_tensors_to_pool[scope].push_back(std::make_tuple(dims_prod, tensor_ptr, from));
    forward_tensors_sent_to_pool[scope].push_back(tensor_ptr);
  }
}



std::map<int, std::map<std::string, std::vector<std::tuple<float, float*,std::string>>>> threaded_tensors_to_pool;
std::map<int, std::map<std::string, std::vector<float*>>> threaded_tensors_sent_to_pool;
std::map<int, std::map<std::string, std::vector<Tensor*>>> threaded_Tensors_to_free;
std::map<int, std::map<std::string, std::vector<float*>>> threaded_tensors_to_save;
std::map<int, std::map<std::string, std::vector<Tensor*>>> threaded_Tensors_to_save;

void to_free_tensor_threaded(Tensor *tensor_ptr, std::string scope, int thread_id)
{
  if(!in_tensor_ptr_vec(tensor_ptr, threaded_Tensors_to_free[thread_id][scope]) && !in_tensor_ptr_vec(tensor_ptr, threaded_Tensors_to_save[thread_id][scope]))
    threaded_Tensors_to_free[thread_id][scope].push_back(tensor_ptr);
}
void to_pool_threaded(float dims_prod, float *tensor_ptr, std::string scope, int thread_id, std::string from)
{
  if (!in_float_ptr_vec(tensor_ptr, threaded_tensors_sent_to_pool[thread_id][scope]) && !in_float_ptr_vec(tensor_ptr, threaded_tensors_to_save[thread_id][scope]))
  {
    threaded_tensors_to_pool[thread_id][scope].push_back(std::make_tuple(dims_prod, tensor_ptr, from));
    threaded_tensors_sent_to_pool[thread_id][scope].push_back(tensor_ptr);
  }
}



void ForwardCleanupToPool(Tensor *back_node, std::string scope);
int DoesTreeContainWeight(Tensor *back_node);
void CleanScopeTensors(std::string scope);




//global
using backward_tuple = std::tuple<int, int, int, int, int, float *, float *, float *, std::string, std::string, std::string>;
std::vector<Tensor *> todo_backward_tensors;

// Tensors
static std::map<std::string, Tensor *> NamedTensorsT;
static std::map<std::string, float *> NamedPinnedTensors;
static std::map<std::string, std::vector<float>> NamedDims;
static std::vector<Tensor> TensorsToDelete;


unsigned char* current_data_attr;
std::vector<float> current_data_attr_dims;


static std::map<std::string, std::string> objectVecs;
static std::map<std::string, int> objectVecsLastId;




extern "C" void PrintDims(std::vector<float> dims)
{
  std::cout << "dims: [";
  for (int i=0; i<dims.size();i++)
  {
    std::cout << (int)dims[i];
    if (i==dims.size()-1)
      std::cout << "]";
    else
      std::cout << ", ";
  }
  std::cout  << "\n";
}
int DimsProd(std::vector<float> dims)
{
  if (dims.size()==1)
    return (int) dims[0];

  float aux=1;
  for (int i = 0; i < dims.size(); i++)
    aux = aux*dims[i];
  return (int)aux;
}

std::vector<float> BatchLessDims(std::vector<float> dims)
{
  // Removes first dim (batch dim).
  if (dims.size()<=1)
    LogError("Cannot remove the batch dimension of a unidimensional tensor.");

  std::vector<float> new_dims;

  for (int i=0; i<dims.size()-1;i++)
    new_dims.push_back(dims[i+1]);

  return new_dims;
}

std::vector<float> RemoveLastDim(std::vector<float> dims)
{
  // Removes first dim (batch dim).
  if (dims.size()<=1)
  {
    return {1.0f};
    //LogError("Cannot remove the batch dimension of a unidimensional tensor.");
  }

  std::vector<float> new_dims;

  for (int i=0; i<dims.size()-1;i++)
    new_dims.push_back(dims[i]);

  return new_dims;
}

std::vector<float> RemoveFirstDim(std::vector<float> dims)
{
  // Removes first dim (batch dim).
  if (dims.size()<=1)
    return {1.0f};

  std::vector<float> new_dims;

  for (int i=0; i<dims.size()-1;i++)
    new_dims.push_back(dims[i+1]);

  return new_dims;
}





extern "C" float *load_img(char *img_name)
{
  int width, height, channels;
  
  unsigned char* image_data = stbi_load(img_name, &width, &height, &channels, 0);

  if (image_data) {
    
    current_data_attr_dims.clear();
    current_data_attr_dims.push_back((float)width);
    current_data_attr_dims.push_back((float)height);
    current_data_attr_dims.push_back((float)channels);


    float *image_data_float = new float[width * height * channels];
  
    // Loop through each pixel and convert to float between 0.0 and 1.0
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
          // Assuming unsigned char has 8 bits, scale by 1/255.0 to get a float value between 0.0 and 1.0
          //image_data_float[(y * width + x) * channels + c] = (float)image_data[(y * width + x) * channels + c] / 255.0f;

          // Convert BHWC into BCHW
          image_data_float[c * (height * width) + y * width + x] = (float)image_data[(y * width + x) * channels + c] / 255.0f;
        }
      }
    }

    stbi_image_free(image_data);
    return image_data_float;
    
  } else {
    std::string img_n = img_name;
    std::string _error = "Failed to open image: " + img_n + ".";
    LogErrorS(_error);
  }

  return nullptr;
}


extern "C" float * gload_img(Tensor tensor, char *img_name, float batch_idx)
{
  //std::cout << "LOADING IMAGE FOR: " << tensor.name <<  "\nImage: " << img_name << "\n";
  

  int width, height, channels;
  
  unsigned char* image_data = stbi_load(img_name, &width, &height, &channels, 0);

  if (image_data) {
    
    std::vector<float> dims = tensor.dims;

    //std::cout << "GLOAD IMG, dims of " << tensor.name << "\n";
    //PrintDims(dims);

    std::vector<float> batchless_dims = BatchLessDims(dims);
    int batchless_dims_prod = DimsProd(batchless_dims);
    
    current_data_attr_dims.clear();
    current_data_attr_dims.push_back((float)width);
    current_data_attr_dims.push_back((float)height);
    current_data_attr_dims.push_back((float)channels);


    if (batchless_dims_prod < width*height*channels)
    {
      std::string t_n = tensor.name;
      std::string _error = "The image dimensions are incompatible with the tensor " + t_n + " dimensions.";
      LogErrorS(_error);


      std::cout << "\nTENSOR BATCHLESS DIMS:" << "\n";
      PrintDims(batchless_dims);

      std::cout << "\nImage required dims: [" << width << ", " << height << ", " << channels << "]\n\n";

      return nullptr;
    }
    if (batch_idx > dims[0])
    {
      std::string _error = "Tried to load a pinned tensor on batch index " + std::to_string(batch_idx) + ", whereas this tensor batch size is " + std::to_string(dims[0]) + ".";
      LogErrorS(_error);
    }



    //float *image_data_float = new float[width * height * channels];
    float *image_data_float = tensor.cpu_tensor_ptr;
    int batch_offset = (int) batchless_dims_prod*batch_idx;
    //std::cout << "batch idx: " << batch_idx << ", batch offset: " << batch_offset << "\n";
  
    // Loop through each pixel and convert to float between 0.0 and 1.0
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
          // Assuming unsigned char has 8 bits, scale by 1/255.0 to get a float value between 0.0 and 1.0
          //image_data_float[batch_offset + (y * width + x) * channels + c] = (float)image_data[(y * width + x) * channels + c] / 255.0f;
          image_data_float[batch_offset + c * (height * width) + y * width + x] = (float)image_data[(y * width + x) * channels + c] / 255.0f;
        }
      }
    }
    stbi_image_free(image_data);

    return image_data_float;
    
  } else {
    std::string img_n = img_name;
    std::string _error = "Failed to open image: " + img_n + ".\n\n";
    LogErrorS(_error);
  }

  return nullptr;
}







extern "C" float * wload_img(Tensor *tensor, char *img_name, float worker_idx, float batch_idx)
{
  //std::cout << "LOADING IMAGE FOR: " << tensor->name <<  "\n";
  //std::cout << "Image: " << img_name <<  "\n";


  int width, height, channels;

  //std::cout << "GLOAD IMG, dims of " << img_name << "\n";
  
  unsigned char* image_data = stbi_load(img_name, &width, &height, &channels, 0);

  if (image_data) {
    
    std::vector<float> dims = tensor->dims;

    
    
    std::vector<float> workerless_dims = BatchLessDims(dims);
    int workerless_dims_prod = DimsProd(workerless_dims);

    std::vector<float> batchless_dims = BatchLessDims(workerless_dims);
    int batchless_dims_prod = DimsProd(batchless_dims);
    
    current_data_attr_dims.clear();
    current_data_attr_dims.push_back((float)width);
    current_data_attr_dims.push_back((float)height);
    current_data_attr_dims.push_back((float)channels);


    if (batchless_dims_prod < width*height*channels)
    {
      std::string t_n = tensor->name;
      std::string _error = "The image dimensions are incompatible with the tensor \033[95m" + t_n + "\033[0m dimensions.";
      LogErrorS(_error);


      std::cout << "\nTENSOR BATCHLESS DIMS:" << "\n";
      PrintDims(batchless_dims);

      std::cout << "\nImage required dims: [" << width << ", " << height << ", " << channels << "]\n\n";
      
      return nullptr;
    }
    if (batch_idx > dims[1])
    {
      std::string _error = "Tried to load a pinned tensor on batch index " + std::to_string(batch_idx) + ", whereas this tensor batch size is " + std::to_string(dims[1]) + ".";
      LogErrorS(_error);
    }



    float *image_data_float = tensor->cpu_tensor_ptr;
    int idx_offset = (int) (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);

    //std::cout << "worker idx: " << worker_idx << ", batch idx: " << batch_idx << ", batch offset: " << idx_offset << "\n";
  
    // Loop through each pixel and convert to float between 0.0 and 1.0
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
          // Assuming unsigned char has 8 bits, scale by 1/255.0 to get a float value between 0.0 and 1.0
          image_data_float[idx_offset + c * (height * width) + y * width + x] = (float)image_data[(y * width + x) * channels + c] / 255.0f;
        }
      }
    }
    stbi_image_free(image_data);

    //std::cout << "returning float image" << "\n";
    return image_data_float;
    
  } else {
    std::string img_n = img_name;
    std::string _error = "Failed to open image: " + img_n + ".\n\n";
    LogErrorS(_error);
  }

  return nullptr;
}




extern "C" float load_bin(Tensor *tensor, char *bin_name)
{

  //std::ifstream file(bin_name, std::ios::binary);
  std::ifstream file(bin_name, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::string _error = bin_name;
    _error = "Failed to open file: " + _error;
    LogErrorS(_error);
    return 0;
  }

  file.seekg(0, std::ios::end);
  std::size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);


  float *file_data = new float[file_size / sizeof(float)];

  file.read(reinterpret_cast<char*>(file_data), file_size);
  file.close();

  //std::cout << bin_name << " has a file size of " << file_size << "\n";
  size_t num_elements = file_size / sizeof(float);


  if (num_elements>tensor->dims_prod)
  {
    std::string _error = "Tried to load binary data of size " + std::to_string(num_elements) + " into tensor " + tensor->name + " of size " + std::to_string(tensor->dims_prod);
    LogErrorS(_error);
    return 0;
  }


  float *image_data_float = tensor->cpu_tensor_ptr;


  for (size_t i = 0; i < num_elements; ++i)
  {
    //std::cout << "" << file_data[i] << "\n";
    image_data_float[i] = file_data[i];
  }
  
  
  delete[] file_data;

  return 0;
}



extern "C" float load_bin_idx(Tensor *tensor, char *bin_name, float first_idx, ...)
{
  std::vector<float> idxs;

  va_list args;
  va_start(args, first_idx);

  idxs.push_back(first_idx);

  for (int i=0; i<10; i++)
  {
    float dim = va_arg(args, float);
    if (dim==TERMINATE_VARARG)
      break;
    idxs.push_back(dim);
  }
  va_end(args);
  



  std::vector<float> dims, aux_dims;
  dims = tensor->dims;
  std::vector<float> new_dims;

  float offset=0;


  if (dims.size()==1)
    new_dims = {1.0f};
  else
  {
    aux_dims = dims;
    for (int i = 0; i < idxs.size(); i++)
    {
      aux_dims = RemoveFirstDim(aux_dims);
      offset += idxs[i]*DimsProd(aux_dims);
    }
    new_dims = aux_dims;
  }

  int new_dims_prod = DimsProd(new_dims);




  //std::ifstream file(bin_name, std::ios::binary);
  std::ifstream file(bin_name, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::string _error = bin_name;
    _error = "Failed to open file: " + _error;
    LogErrorS(_error);
    return 0;
  }

  file.seekg(0, std::ios::end);
  std::size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);


  float *file_data = new float[file_size / sizeof(float)];

  file.read(reinterpret_cast<char*>(file_data), file_size);
  file.close();

  //std::cout << bin_name << " has a file size of " << file_size << "\n";
  size_t num_elements = file_size / sizeof(float);


  if (num_elements>tensor->dims_prod)
  {
    std::string _error = "Tried to load binary data of size " + std::to_string(num_elements) + " into tensor " + tensor->name + " of size " + std::to_string(tensor->dims_prod);
    LogErrorS(_error);
    return 0;
  }


  float *image_data_float = tensor->cpu_tensor_ptr + (int) offset;


  for (size_t i = 0; i < num_elements; ++i)
    image_data_float[i] = file_data[i];
  
  
  
  delete[] file_data;

  return 0;
}




extern "C" float wload_bin(Tensor *tensor, char *bin_name, float worker_idx, float batch_idx)
{
  //std::cout << "LOADING BINARY FOR: " << tensor->name <<  "\n";
  //std::cout << "Binary: " << bin_name <<  "\n";

  std::ifstream file(bin_name, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::string _error = bin_name;
    _error = "Failed to open file: " + _error;
    LogErrorS(_error);
    return 0;
  }

  file.seekg(0, std::ios::end);
  std::size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);


  float *file_data = new float[file_size / sizeof(float)];

  // Read the binary file into the tensor_data pointer
  file.read(reinterpret_cast<char*>(file_data), file_size);
  file.close();

  size_t num_elements = file_size / sizeof(float);





  std::vector<float> dims = tensor->dims;

  std::vector<float> workerless_dims = BatchLessDims(dims);
  int workerless_dims_prod = DimsProd(workerless_dims);

  std::vector<float> batchless_dims = BatchLessDims(workerless_dims);
  int batchless_dims_prod = DimsProd(batchless_dims);


  float *image_data_float = tensor->cpu_tensor_ptr;
  int idx_offset = (int) (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);


  //todo: add out of bounds error here



  for (size_t i = 0; i < num_elements; ++i)
  {
    //std::cout << ", " << file_data[i];
    image_data_float[idx_offset + i] = file_data[i];
  }
  
  delete[] file_data;

  return 0;
}


extern "C" float save_as_bin(int thread_id, Tensor *tensor, char *bin_name)
{
  std::ofstream file(bin_name, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << bin_name << "\n";
    return -1;
  }


  if (tensor->cpu_tensor_ptr==nullptr)
    cudaMallocHost(&tensor->cpu_tensor_ptr, tensor->dims_prod*sizeof(float));
  
  cudaMemcpy(tensor->cpu_tensor_ptr, tensor->tensor_ptr, tensor->dims_prod*sizeof(float), cudaMemcpyDeviceToHost);

  
  file.write(reinterpret_cast<const char*>(tensor->cpu_tensor_ptr), tensor->dims_prod * sizeof(float));
  file.close();


  return 0;
}



extern "C" float save_as_int(int thread_id, Tensor *tensor, char *bin_name)
{
  // Open the binary file for writing
  std::ofstream file(bin_name, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << bin_name << "\n";
    return -1;
  }


  if (tensor->cpu_tensor_ptr==nullptr)
    cudaMallocHost(&tensor->cpu_tensor_ptr, tensor->dims_prod*sizeof(float));
  
  cudaMemcpy(tensor->cpu_tensor_ptr, tensor->tensor_ptr, tensor->dims_prod*sizeof(float), cudaMemcpyDeviceToHost);

  int *int_data = new int[tensor->dims_prod];

  for (int i=0; i < tensor->dims_prod; ++i)
    int_data[i] = (int)tensor->cpu_tensor_ptr[i];

  
  file.write(reinterpret_cast<const char*>(int_data), tensor->dims_prod * sizeof(float));
  file.close();


  return 0;
}




unsigned char *interpolate_img(unsigned char *src, int height, int width, int dst_height, int dst_width)
{
    unsigned char *new_img = new unsigned char[dst_height*dst_width*3];


    // Calculate scale factors for width and height
    float xScale = static_cast<float>(width) / dst_width;
    float yScale = static_cast<float>(height) / dst_height;



    for (int y = 0; y < dst_height; ++y) {
        for (int x = 0; x < dst_width; ++x) {
            // Find the corresponding position in the input image
            float srcX = x * xScale;
            float srcY = y * yScale;
            int x1 = static_cast<int>(srcX);
            int y1 = static_cast<int>(srcY);
            int x2 = std::min(x1 + 1, width - 1);
            int y2 = std::min(y1 + 1, height - 1);

            float alphaX = srcX - x1;
            float alphaY = srcY - y1;

            // Get the pixel values from the input image
            auto getPixel = [&](int x, int y, int c) {
                int index = (y * width + x) * 3 + c;
                return static_cast<float>(src[index]);
            };

            // Perform bilinear interpolation
            for (int c = 0; c < 3; ++c) {
                float value1 = (1 - alphaX) * getPixel(x1, y1, c) + alphaX * getPixel(x2, y1, c);
                float value2 = (1 - alphaX) * getPixel(x1, y2, c) + alphaX * getPixel(x2, y2, c);
                float interpolatedValue = (1 - alphaY) * value1 + alphaY * value2;
                new_img[(y * dst_width + x) * 3 + c] = static_cast<unsigned char>(std::clamp(interpolatedValue, 0.0f, 255.0f));
            }
        }
    }

  delete[] src;
  return new_img;
}


uint uint_min(uint x, int y){
  uint _y = (uint) y;
  if (x>_y)
    return _y;
  return x;
}

double lerp_bli(double c1, double c2, double v1, double v2, double x)
{
  if( (v1==v2) ) return c1;
  double inc = ((c2-c1)/(v2 - v1)) * (x - v1);
  double val = c1 + inc;
  return val;
};

unsigned char *bilinear_resize(unsigned char *src, int height, int width, int dst_height, int dst_width)
{
    unsigned char *new_img = new unsigned char[dst_height*dst_width*3];
    std::memset(new_img, 0, dst_height * dst_width * 3 * sizeof(unsigned char));

    // x and y ratios
    double rx = (double)width / (double)dst_width;
    double ry = (double)height / (double)dst_height;


    
    // loop through destination image
    for(int y=0; y<dst_height; ++y)
    {
        for(int x=0; x<dst_width; ++x)
        {
            //double sx = x * rx;
            //double sy = y * ry;
            double sx = (width>dst_width) ? (x + 0.5) * rx - 0.5 : x * rx;
            double sy = (height>dst_height) ? (y + 0.5) * ry - 0.5 : y * ry;
            
            
            uint xl = std::floor(sx);
            uint yt = std::floor(sy);
            uint xr = (width>dst_width) ? xl+1 : xl;
            uint yb = (height>dst_height) ? yt+1 : yt;
            //uint xr = uint_min(xl + 1, width - 1);
            //uint yb = uint_min(yt + 1, height - 1);


            if (height<dst_height)
              std::cout << yt << ", " << yb << ", " << y << ", " << sy << ". " << height << ", " << dst_height << "\n";

            
            for (uint d = 0; d < 3; ++d)
            {
                unsigned char tl    = src[(xl*width+yt)*3+d];//GetData(xl, yt, d);
                unsigned char tr    = src[(xr*width+yt)*3+d];//GetData(xr, yt, d);
                unsigned char bl    = src[(xl*width+yb)*3+d];//GetData(xl, yb, d);
                unsigned char br    = src[(xr*width+yb)*3+d];//GetData(xr, yb, d);
                double t    = lerp_bli(tl, tr, xl, xr, sx);
                double b    = lerp_bli(bl, br, xl, xr, sx);
                double m    = lerp_bli(t, b, yt, yb, sy);
                unsigned char val   = std::floor(m + 0.5);
                
                new_img[(x * dst_width + y) * 3 + d] = val;
            }
        }
    }

  delete[] src;
  return new_img;
}

extern "C" float * wload_img_resize(Tensor *tensor, char *img_name, float worker_idx, float batch_idx, float c, float h, float w)
{
  //std::cout << "LOADING IMAGE FOR: " << tensor->name <<  "\n";
  //std::cout << "Image: " << img_name <<  "\n";


  int width, height, channels;

  //std::cout << "GLOAD IMG, dims of " << img_name << "\n";
  
  unsigned char* image_data = stbi_load(img_name, &width, &height, &channels, 0);



  if(width!=w||height!=h)
  { 
    image_data = interpolate_img(image_data, height, width, h, w);
    width = w;
    height = h;
  }



  if (image_data) {
    
    std::vector<float> dims = tensor->dims;

    
    
    std::vector<float> workerless_dims = BatchLessDims(dims);
    int workerless_dims_prod = DimsProd(workerless_dims);

    std::vector<float> batchless_dims = BatchLessDims(workerless_dims);
    int batchless_dims_prod = DimsProd(batchless_dims);
    
    current_data_attr_dims.clear();
    current_data_attr_dims.push_back((float)width);
    current_data_attr_dims.push_back((float)height);
    current_data_attr_dims.push_back((float)channels);


    if (batchless_dims_prod < width*height*channels)
    {
      std::string t_n = tensor->name;
      std::string _error = "The image dimensions are incompatible with the tensor \033[95m" + t_n + "\033[0m dimensions.";
      LogErrorS(_error);


      std::cout << "\nTENSOR BATCHLESS DIMS:" << "\n";
      PrintDims(batchless_dims);

      std::cout << "\nImage required dims: [" << width << ", " << height << ", " << channels << "]\n\n";
      
      return nullptr;
    }
    if (batch_idx > dims[1])
    {
      std::string _error = "Tried to load a pinned tensor on batch index " + std::to_string(batch_idx) + ", whereas this tensor batch size is " + std::to_string(dims[1]) + ".";
      LogErrorS(_error);
    }



    float *image_data_float = tensor->cpu_tensor_ptr;
    int idx_offset = (int) (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);

    //std::cout << "worker idx: " << worker_idx << ", batch idx: " << batch_idx << ", batch offset: " << idx_offset << "\n";
  
    // Loop through each pixel and convert to float between 0.0 and 1.0
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
          // Assuming unsigned char has 8 bits, scale by 1/255.0 to get a float value between 0.0 and 1.0
          image_data_float[idx_offset + c * (height * width) + y * width + x] = (float)image_data[(y * width + x) * channels + c] / 255.0f;
        }
      }
    }
    stbi_image_free(image_data);

    //std::cout << "returning float image" << "\n";
    return image_data_float;
    
  } else {
    std::string img_n = img_name;
    std::string _error = "Failed to open image: " + img_n + ".\n\n";
    LogErrorS(_error);
  }

  return nullptr;
}


extern "C" float save_img(int thread_id, Tensor *tensor, char *img_name)
{
  
  int c, h, w;

  c = tensor->dims[tensor->dims.size()-3];
  h = tensor->dims[tensor->dims.size()-2];
  w = tensor->dims[tensor->dims.size()-1];


  // Check if the tensor has 3 channels
  if (c != 3) {
    std::cerr << "Only 3-channel images are supported." << std::endl;
    return -1;
  }

  float *tensor_cpu = new float[c*h*w];
  cudaCheck(cudaMemcpy(tensor_cpu, tensor->tensor_ptr, c*h*w*sizeof(float), cudaMemcpyDeviceToHost));

  // Convert tensor to 8-bit format
  std::vector<uint8_t> imageData(h * w * c);
  for (int ch = 0; ch < c; ++ch) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        int tensorIndex = (ch * h + y) * w + x;
        int imageIndex = (y * w + x) * c + ch;
        imageData[imageIndex] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, tensor_cpu[tensorIndex] * 255.0f)));
      }
    }
  }

  std::string img = "/home/nosaveddata/imgs/";
  img = img + img_name;
  img = img + RandomString(4);
  img = img + ".png";

  std::cout << "writing image: " << img << "\n";

  // Write the image using stb_image_write
  if (!stbi_write_png(img.c_str(), w, h, c, imageData.data(), w * c)) {
    std::cerr << "Failed to write image to " << img_name << std::endl;
  }

  delete[] tensor_cpu;

  return 0;
}

extern "C" float cpu(int thread_id, Tensor *tensor)
{

  float *tensor_ptr, *tensor_cpu;
  tensor_ptr = tensor->tensor_ptr;
  tensor_cpu = tensor->cpu_tensor_ptr;

  cudaStream_t stream = ThreadsStream[thread_id];
  cudaStreamSynchronize(stream);

  if (tensor_ptr==nullptr)
    LogErrorS("Cannot load tensor to cpu from an null tensor.");

  if (tensor_cpu!=nullptr)
    cudaCheck(cudaFree(tensor_cpu));

  float dims_prod = tensor->dims_prod;



  cudaMallocHost(&tensor_cpu, dims_prod*sizeof(float));
  cudaMemcpy(tensor_cpu, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToHost);

  tensor->cpu_tensor_ptr = tensor_cpu;


  return 0;
}

extern "C" float cpu_idx(Tensor *tensor, float idx)
{

  float *tensor_cpu;
  tensor_cpu = tensor->cpu_tensor_ptr;


  if (tensor_cpu==nullptr)
    LogErrorS("Cannot idx a null cpu tensor.");

  float dims_prod = tensor->dims_prod;
  if (idx>dims_prod)
    LogErrorS("Idx higher than dims prod at cpu_idx().");

  

  return tensor_cpu[(int)idx];
}



std::map<std::string, int> Vocab;
float max_tokens;
float last_tok_id = 2;


void ProcessString(std::string& str) {
  // to lower and remove ponctuation


  // Convert to lowercase
  std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c) { return std::tolower(c); });

  // Remove punctuation
  str.erase(std::remove_if(str.begin(), str.end(),
                   [](unsigned char c) { return std::ispunct(c); }),
            str.end());
}


extern "C" float build_vocab(char *filename, float _max_tokens)
{
  pthread_mutex_lock(&vocab_mutex); // Files are not thread safe
  std::ifstream file(filename);
  max_tokens = _max_tokens;

  if (!file) {
    std::cerr << "Error opening file " << filename << std::endl;
    return 1;
  }

  std::string line;
  while (std::getline(file, line)) 
  {       
    std::istringstream lineStream(line); // Create a string stream from the line
    std::string word;

    while (lineStream >> word) {
      ProcessString(word);
      
      if (Vocab.count(word)==0 && last_tok_id<(max_tokens-2)) // -2 for padding and unk tokens
      {
        Vocab[word] = last_tok_id;
        last_tok_id+=1;
      }
    }
  }

  file.close();
  pthread_mutex_unlock(&vocab_mutex);

  return 0;
}


extern "C" float tokenize(Tensor *tensor, char *filename)
{
  pthread_mutex_lock(&vocab_mutex); // Files are not thread safe
  std::ifstream file(filename);

  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return 1;
  }

  std::string line;
  while (std::getline(file, line)) 
  {       
    std::istringstream lineStream(line); // Create a string stream from the line
    std::string word;

    while (lineStream >> word) {

      ProcessString(word);

      int idx;

      std::cout << word << "\n";
      idx = (Vocab.count(word)>0) ? Vocab[word] : UNK_TOK;
      

      std::cout << idx << std::endl;
    }
  }

  file.close();
  pthread_mutex_unlock(&vocab_mutex);

  return 0;
}


extern "C" float wtokenize(Tensor *tensor, char *filename, float trunc_to, float worker_idx, float batch_idx)
{
  //tensor e [workers, seq_len, batch_size, vocab_size]

  //pthread_mutex_lock(&vocab_mutex); // Files are not trhead safe
  std::ifstream file(filename);

  //std::cout << "Loading" <<  filename << "\n";
  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return 1;
  }


  std::vector<float> dims = tensor->dims;
  float dims_prod = tensor->dims_prod;


  std::vector<float> workerless_dims = BatchLessDims(dims);
  int workerless_dims_prod = DimsProd(workerless_dims);

  std::vector<float> seqless_dims = BatchLessDims(workerless_dims);
  int seqless_dims_prod = DimsProd(seqless_dims);

  std::vector<float> batchless_dims = BatchLessDims(seqless_dims);
  int batchless_dims_prod = DimsProd(batchless_dims);

  int idx_offset = (int) (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);


  //TODO: add pad and left

  std::string line;
  int words_count = 0;
  while (std::getline(file, line)) 
  {       
    std::istringstream lineStream(line); // Create a string stream from the line
    std::string word;

    while (lineStream >> word)
    {
      words_count++;
      if (words_count>trunc_to)
        break;

      ProcessString(word);

      int idx, pre_idx;

      idx = (Vocab.count(word)>0) ? Vocab[word] : UNK_TOK; // inplace onehot
      idx = (idx<0) ? 0 : idx;


      pre_idx = idx;
      idx = idx + idx_offset;

      if(idx>dims_prod)
      {
        std::string _err = "Tokenizer IDX " + std::to_string(idx) + " is higher than allowed dims: " +std::to_string(dims_prod) + " from pre-idx: " + std::to_string(pre_idx);
        LogErrorS(_err);
      }

      tensor->cpu_tensor_ptr[idx] = 1;

      idx_offset += seqless_dims_prod; //moves to the next sequence element
      /*
      if(idx_offset>dims_prod)
      {
        std::string _err = "Tokenizer Index " + std::to_string(idx_offset) + " is higher than allowed dims: " +std::to_string(dims_prod);
        LogErrorS(_err);
      }
      */
    }
    if (words_count>trunc_to)
      break;
  }

  file.close();
  //pthread_mutex_unlock(&vocab_mutex);

  return 0;
}


extern "C" float wtokenize_pad_left(Tensor *tensor, char *filename, float trunc_to, float worker_idx, float batch_idx)
{
  // x e [W, T, B]
  
  std::ifstream file(filename);

  //std::cout << "Loading" <<  filename << "\n";
  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return 1;
  }


  std::vector<float> dims = tensor->dims;
  float dims_prod = tensor->dims_prod;


  std::vector<float> workerless_dims = BatchLessDims(dims);
  int workerless_dims_prod = DimsProd(workerless_dims);

  std::vector<float> seqless_dims = BatchLessDims(workerless_dims);
  int seqless_dims_prod = DimsProd(seqless_dims);

  std::vector<float> batchless_dims = BatchLessDims(seqless_dims);
  int batchless_dims_prod = DimsProd(batchless_dims);

  int idx_offset = (int) (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);



  int *indices  = new int[trunc_to];
  int *padded_indices  = new int[trunc_to];

  //for (int i=0; i<tensor->dims_prod; i++)
  //  tensor->cpu_tensor_ptr[i] = 0.0f;

  for (int i = 0; i < trunc_to; i++)
    padded_indices[i] = PAD_TOK;


  std::string line;
  int words_count = 0;
  while (std::getline(file, line)) 
  {       
    std::istringstream lineStream(line); // Create a string stream from the line
    std::string word;

    while (lineStream >> word)
    {
      words_count++;
      if (words_count>trunc_to)
        break;

      ProcessString(word);

      int idx;

      idx = (Vocab.count(word)>0) ? Vocab[word] : UNK_TOK; // inplace onehot
      idx = (idx<0) ? 0 : idx;

      indices[words_count-1] = idx;      
    }
    if (words_count>trunc_to)
      break;
  }
  file.close();


  if (words_count<=trunc_to)
  {
    int offset = trunc_to-words_count;
    for(int i=0; i<words_count; i++)
    {
      padded_indices[i+offset] = indices[i];
    }
  } else 
    padded_indices = indices;


  int idx;
  for(int i=0; i<trunc_to; i++)
  {
    idx = padded_indices[i] + idx_offset;

    tensor->cpu_tensor_ptr[idx] = 1;

    idx_offset += seqless_dims_prod; //moves to the next sequence element
  }
  
  /*
  std::cout << "[";
  for (int i = 0; i < trunc_to; i++)
    std::cout << padded_indices[i] << ", ";
  std::cout << "]" << "\n\n";  
  */
  

  delete[] indices;
  if (words_count<=trunc_to)
    delete[] padded_indices;

  return 0;
}



extern "C" float wtokenize_pad_left_batch_first(Tensor *tensor, char *filename, float trunc_to, float worker_idx, float batch_idx)
{
  // x e [W, B, T, V]
  

  std::ifstream file(filename);

  //std::cout << "Loading" <<  filename << "\n";
  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return 1;
  }


  std::vector<float> dims = tensor->dims;
  float dims_prod = tensor->dims_prod;


  std::vector<float> workerless_dims = BatchLessDims(dims);
  int workerless_dims_prod = DimsProd(workerless_dims);

  std::vector<float> batchless_dims = BatchLessDims(workerless_dims);
  int batchless_dims_prod = DimsProd(batchless_dims);

  std::vector<float> seqless_dims = BatchLessDims(batchless_dims);
  int seqless_dims_prod = DimsProd(seqless_dims);



  int idx_offset = (int) (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);



  int *indices  = new int[trunc_to];
  int *padded_indices  = new int[trunc_to];

  //for (int i=0; i<tensor->dims_prod; i++)
  //  tensor->cpu_tensor_ptr[i] = 0.0f;

  for (int i = 0; i < trunc_to; i++)
    padded_indices[i] = PAD_TOK;


  std::string line;
  int words_count = 0;
  while (std::getline(file, line)) 
  {       
    std::istringstream lineStream(line); // Create a string stream from the line
    std::string word;

    while (lineStream >> word)
    {
      words_count++;
      if (words_count>trunc_to)
        break;

      ProcessString(word);

      int idx;

      idx = (Vocab.count(word)>0) ? Vocab[word] : UNK_TOK; // inplace onehot
      idx = (idx<0) ? 0 : idx;

      indices[words_count-1] = idx;      
    }
    if (words_count>trunc_to)
      break;
  }
  file.close();



  // pad indices
  if (words_count<=trunc_to)
  {
    int offset = trunc_to-words_count;
    for(int i=0; i<words_count; i++)
    {
      padded_indices[i+offset] = indices[i];
    }
  } else 
    padded_indices = indices;



  // one-hot and save it into the tensor
  int idx;
  for(int i=0; i<trunc_to; i++)
  {
    idx = padded_indices[i] + idx_offset;

    tensor->cpu_tensor_ptr[idx] = 1;

    idx_offset += seqless_dims_prod; //moves to the next sequence element
  }

  
  
  delete[] indices;
  if (words_count<=trunc_to)
    delete[] padded_indices;

  return 0;
}


extern "C" float wtokenize_pad_left_idx(Tensor *tensor, char *filename, float trunc_to, float worker_idx, float batch_idx)
{
  // x e [W, B, T]
  

  std::ifstream file(filename);

  //std::cout << "Loading" <<  filename << "\n";
  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return 1;
  }


  std::vector<float> dims = tensor->dims;
  float dims_prod = tensor->dims_prod;


  std::vector<float> workerless_dims = BatchLessDims(dims);
  int workerless_dims_prod = DimsProd(workerless_dims);

  std::vector<float> batchless_dims = BatchLessDims(workerless_dims);
  int batchless_dims_prod = DimsProd(batchless_dims);

  



  int idx_offset = (int) (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);



  int *indices  = new int[trunc_to];
  int *padded_indices  = new int[trunc_to];

  //for (int i=0; i<tensor->dims_prod; i++)
  //  tensor->cpu_tensor_ptr[i] = 0.0f;

  for (int i = 0; i < trunc_to; i++)
    padded_indices[i] = PAD_TOK;


  std::string line;
  int words_count = 0;
  while (std::getline(file, line)) 
  {       
    std::istringstream lineStream(line); // Create a string stream from the line
    std::string word;

    while (lineStream >> word)
    {
      words_count++;
      if (words_count>trunc_to)
        break;

      ProcessString(word);

      int idx;

      idx = (Vocab.count(word)>0) ? Vocab[word] : UNK_TOK; // inplace onehot
      idx = (idx<0) ? 0 : idx;

      indices[words_count-1] = idx;      
    }
    if (words_count>trunc_to)
      break;
  }
  file.close();



  // pad indices
  if (words_count<=trunc_to)
  {
    int offset = trunc_to-words_count;
    for(int i=0; i<words_count; i++)
    {
      padded_indices[i+offset] = indices[i];
    }
  } else 
    padded_indices = indices;




  //std::cout << "[";
  // one-hot and save it into the tensor
  int idx = idx_offset;
  for(int i=0; i<trunc_to; i++)
  {
    tensor->cpu_tensor_ptr[idx] = padded_indices[i];

    //std::cout << padded_indices[i] << ",";

    idx += 1; //moves to the next sequence element
  }
  //std::cout << "]\n";
  
  
  delete[] indices;
  if (words_count<=trunc_to)
    delete[] padded_indices;

  return 0;
}



extern "C" float write_zerosw(Tensor *tensor, float worker_idx)
{
  std::vector<float> dims = tensor->dims;

  std::vector<float> workerless_dims = BatchLessDims(dims);
  int workerless_dims_prod = DimsProd(workerless_dims);

  int idx_offset = (int) (workerless_dims_prod*worker_idx);

  for(int i=0; i<workerless_dims_prod; i++)
    tensor->cpu_tensor_ptr[i+idx_offset] = 0.0f;
  
  return 0;
}


extern "C" float *split_str_to_float(char *in_string, int gather_position)
{
  std::vector<std::string> splitted = split_str(in_string, '/');

  float * ret = new float[1];

  if(gather_position<0)
    gather_position = splitted.size()+gather_position;

  ret[0] = std::stof(splitted[gather_position]);

  return ret;
}


extern "C" void *to_string(float v)
{
  //todo: allow float instead of int only
  return str_to_char(std::to_string((int)v));
}


extern "C" void *cat_str_float(char *c, float v)
{

  std::string s = c;
  std::string tmp = std::to_string((int)v);

  s = s + c;

  return str_to_char(s);
}



//
static std::unique_ptr<ExprAST> ParseSelfExpr(std::string class_name="") {

  std::string pre_dot = "";
  std::string type = "None";
  std::string object_class;
  bool is_class_attr=false;
  bool is_self=false;
  bool is_vec=false;
  std::vector<std::tuple<std::string, int, std::vector<std::unique_ptr<ExprAST>>>> Names;
  std::string IdName;



  if (CurTok!=tok_self)
  {
    is_class_attr = true;
    pre_dot="";
  } else 
  {
    is_self=true;
    getNextToken(); // eat self
    Names.push_back(std::make_tuple("_", type_self, std::vector<std::unique_ptr<ExprAST>>{}));
  }


  
  int i=0;
  while (CurTok==tok_identifier||CurTok==tok_class_attr||CurTok==tok_post_class_attr_attr||CurTok==tok_post_class_attr_identifier)
  {
    int _type = type_attr;
    
    if (in_str(IdentifierStr, objectVars))
    {
      
      //std::cout << "\n\n" << IdentifierStr << " IS ON OBJECT VARS" <<  "\n\n";
      _type = type_object_name;
    }
      
    if (i==0&&CurTok==(tok_class_attr))
    {
      Names.push_back(std::make_tuple(IdentifierStr, _type, std::vector<std::unique_ptr<ExprAST>>{}));
      _type = type_attr;
    }
      
    if (i>0)
      is_class_attr = true;

    if (CurTok!=tok_identifier&&CurTok!=tok_post_class_attr_identifier)
    {
      object_class=IdentifierStr;
      pre_dot+=IdentifierStr;
    }

    is_vec=false;
    if (CurTok==tok_identifier||CurTok==tok_post_class_attr_identifier) // Need to handle vector
      IdName = IdentifierStr;

    //std::cout << "\n\ntok pre vec: " << ReverseToken(CurTok) << "\n";

    getNextToken(); // eat attr/identifier

      
    if (CurTok=='[')
    {
      //std::cout << "tokvec: " << ReverseToken(CurTok) << "\n";
      getNextToken(); // eat [
      std::vector<std::unique_ptr<ExprAST>> idx = ParseIdx(class_name);
      getNextToken(); // eat ]
      _type = type_vec;
      //int _type = (Object_toClassVec.count(IdName)>0) ? type_object_vec : type_vec;
      Names.push_back(std::make_tuple(IdName, _type, std::move(idx)));
      is_vec=true;
    } else
      Names.push_back(std::make_tuple(IdName, _type, std::vector<std::unique_ptr<ExprAST>>{}));

    //std::cout << "tok: " << ReverseToken(CurTok) << "\n";
    i+=1;
  }
    

  
  //std::cout << "Post tok: " << ReverseToken(CurTok) << "\n";
  
  // Turns string from object model of class type Model into Model
  if (Object_toClass.count(object_class)>0)
    object_class = Object_toClass[object_class]; 
  
  
  
  
  //std::cout << "\n\nParseSelfExpr of " << IdentifierStr << " HAS CLASS: " << class_name << " and pre-dot: " << pre_dot << "\n\n\n";



  

  if (!is_vec&&CurTok!='(') // Simple variable ref.
  {
    //std::cout << "Parsing a var" << "\n";
    
    if (in_str(IdName, pinnedTensorVars))
      type = "pinned_tensor";
    if (in_str(IdName, tensorVars))
      type = "tensor";
    if (in_str(IdName, objectVars))
      type = "object";
    if (functionVars.find(IdName) != functionVars.end())
      type = "tensor";
    if (stringMethods.find(IdName) != stringMethods.end())
      type = "str";
    if (in_str(IdName, floatVars))
      type = "float";
    if (in_str(IdName, float_vecVars))
      type = "float_vec";
    if (in_str(IdName, strVars))
      type = "str";
    if (in_str(IdName, str_vecVars))
      type = "str_vec";

    std::cout << "Var type: " << type << "\n";

    auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
    auto aux = std::make_unique<VariableExprAST>(std::move(name_solver_expr), type);

    aux->SetSelf(is_self);
    aux->SetIsAttribute(is_class_attr);
    aux->SetPreDot(pre_dot);
    
    
    if (CurTok==tok_space)
      getNextToken();

    return aux;
  }



  if (is_vec)
  {
    std::unique_ptr<ExprAST> aux;
    std::vector<std::unique_ptr<ExprAST>> Idx;
    
    
    
    if (in_str(IdName, str_vecVars))
      type= "str_vec";
    if (in_str(IdName, float_vecVars))
      type = "float_vec";
    if (in_str(IdName, pinnedTensorVars))
      type = "pinned_tensor";
    if (in_str(IdName, tensorVars))
      type= "tensor";
    if (in_str(IdName, objectVars))
      type = "object_vec";

    if(type=="object_vec")
      Idx = std::vector<std::unique_ptr<ExprAST>>{};
    else
      Idx = std::move(std::get<2>(Names[Names.size()-1]));

    auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
    aux = std::make_unique<VecIdxExprAST>(std::move(name_solver_expr), std::move(Idx), type);
    aux->SetIsVec(true);


    aux->SetSelf(is_self);
    aux->SetIsAttribute(is_class_attr);
    aux->SetPreDot(pre_dot);


    return std::move(aux);
  }







  // PARSE CALL.

  getNextToken(); // eat (
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (true) {
      if (auto Arg = ParseExpression(class_name))
      {
        //std::cout << "Parsed arg " << Arg->GetName() << "\n";
        Args.push_back(std::move(Arg));
      }
        
      else
        return nullptr;

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        return LogError("Esperado ')' ou ',' na lista de argumentos");
      getNextToken();
    }
  }

  
  // varargs
  if (in_str(IdName, vararg_methods))
    Args.push_back(std::make_unique<NumberExprAST>(TERMINATE_VARARG));
  
  

  // Eat the ')'.
  getNextToken();




  // Override function calls: e.g: conv1 -> Conv2d
  bool is_var_forward = false;
  bool return_tensor = false;
  bool return_string = false;
  std::string callee_override = "none";
  if (functionVars.count(IdName) > 0)
  {
    is_var_forward = true;
    return_tensor = true;
    callee_override = functionVars[IdName];

  } else if (floatFunctions.find(IdName) != floatFunctions.end())
  {
    is_var_forward = true;
    callee_override = floatFunctions[IdName];

  } else if (stringMethods.find(IdName) != stringMethods.end())
  {
    is_var_forward = true;
    return_string = true;
    callee_override = stringMethods[IdName];

  } else {
    if (is_self && !is_class_attr)
      IdName = class_name + IdName;
  }

  
  //std::cout << "\nCalling method: " << IdName << " for pre-dot: " << pre_dot << "\n\n";

  //if (IdName == "len")


  auto name_solver_expr = std::make_unique<NameSolverAST>(std::move(Names));
  name_solver_expr->SetNameSolveToLast(false);
  auto aux = std::make_unique<CallExprAST>(std::move(name_solver_expr), IdName, std::move(Args),
                                        object_class, pre_dot, is_var_forward, callee_override);

  if (in_str(IdName, return_tensor_fn) || return_tensor)
    aux->SetType("tensor");
  
  if (return_string||in_str(IdName, return_string_fn))
    aux->SetType("str");

  if (CurTok==tok_space)
    getNextToken();

  if (is_self)
    aux->SetSelf(true);    
  if (is_class_attr)
    aux->SetIsAttribute(true);
  aux->SetPreDot(pre_dot);
  
  return aux;
}


//
static std::unique_ptr<ExprAST> ParsePinnedTensorExpr() {
  
  getNextToken(); // eat pinned_tensor.
  
  if (CurTok != '[')
    return LogError("pinned tensor declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "zeros";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected a number or var on the tensor dimension.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("A tensor's dimension should be int, not float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  
  if (CurTok != ']')
    return LogError("] not found.");
    getNextToken();


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected pinned tensor identifier name.");

  while (true) {
    std::string Name = IdentifierStr;
    pinnedTensorVars.push_back(IdentifierStr);
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected pinned tensor identifier names.");
  }



  auto aux = std::make_unique<PinnedTensorExprAST>(std::move(VarNames), "pinned_tensor",
                                             std::move(dims), init);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}





//
static std::unique_ptr<ExprAST> ParseTensorExpr(std::string class_name="") {
  bool is_weight;
  if (CurTok==tok_tensor)
    is_weight=false;
  if (CurTok==tok_param)
    is_weight=true;

  getNextToken(); // eat the tensor.

  
  if (CurTok != '[')
    return LogError("tensor declaration expected [");
  getNextToken();
  
  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "xavu_relu";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected a number or var on the tensor dimension.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("A tensor's dimension should be int, not float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      //dims.push_back(std::move(ParseExpression(class_name)));
      dims.push_back(std::move(ParsePrimary(class_name)));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  
  if (CurTok != ']')
    return LogError("] not found at tensor declaration.");
    getNextToken();



  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true; //TODO: set self per VarName instead.
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected tensor identifier name.");

  while (true) {
    std::string Name = IdentifierStr;
    tensorVars.push_back(IdentifierStr);
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init))); 

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected tensor identifier names.");
  }



  auto aux = std::make_unique<TensorExprAST>(std::move(VarNames), "tensor",
                                             std::move(dims), init, is_weight);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);
  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}




//
static std::unique_ptr<ExprAST> ParseConv2dExpr() {
  
  getNextToken(); // eat the Conv2d.
  
  if (CurTok != '[')
    return LogError("Conv2d declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "xavu_relu";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  

  if (CurTok != ']')
    return LogError("Expected ].");
    getNextToken();

  if (dims.size()<5)
    return LogError("Convolution declaration requires input and output channels, kernel size, stride and padding.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected conv identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "ConvForward2d";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Esperado um ou mais identificadores após var.");
  }



  auto aux = std::make_unique<Conv2dExprAST>(std::move(VarNames), "conv2d",
                                             std::move(dims[0]), std::move(dims[1]), std::move(dims[2]),
                                             std::move(dims[3]), std::move(dims[4]),
                                             init);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}



//
static std::unique_ptr<ExprAST> ParseLSTMExpr() {
  
  getNextToken(); // eat the LSTM.
  
  if (CurTok != '[')
    return LogError("LSTM declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "xavu_relu";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  

  if (CurTok != ']')
    return LogError("Expected ].");
    getNextToken();

  if (dims.size()!=2 && dims.size()!=3)
    return LogError("LSTM requires input and output hiddens count.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected LSTM identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "LSTMForward";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers after LSTM.");
  }



  auto aux = std::make_unique<LSTMExprAST>(std::move(VarNames), "lstm",
                                             std::move(dims[0]), std::move(dims[1]),
                                             init);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}


static std::unique_ptr<ExprAST> ParseEmbeddingExpr() {
  
  getNextToken(); // eat the Embedding.
  
  if (CurTok != '[')
    return LogError("Embedding declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "xavu_relu";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  

  if (CurTok != ']')
    return LogError("Expected ].");
    getNextToken();

  if (dims.size()!=2 && dims.size()!=3)
    return LogError("Embedding requires input vocabulary size and output hiddens count.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected Embedding identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "EmbeddingForward";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers after Embedding.");
  }



  auto aux = std::make_unique<EmbeddingExprAST>(std::move(VarNames), "embedding",
                                             std::move(dims[0]), std::move(dims[1]),
                                             init);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}




static std::unique_ptr<ExprAST> ParseLinearExpr() {
  
  getNextToken(); // eat the Linear.
  
  if (CurTok != '[')
    return LogError("Linear declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "xavu";
  std::vector<int> notators_vec;
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else if (in_str(IdentifierStr, notators_str)){
        notators_vec.push_back(NotatorsMap[IdentifierStr]);
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }


  if (CurTok != ']')
    return LogError("Expected ] at Linear.");
    getNextToken();

  if (dims.size()<2)
    return LogError("Linear requires input and output dimensions.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected Linear identifier name.");


  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "LinearForward";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers after Linear.");
  }


  auto aux = std::make_unique<LinearExprAST>(std::move(VarNames), "linear",
                                             std::move(dims[0]), std::move(dims[1]),
                                             std::move(notators_vec),
                                             init);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}







static std::unique_ptr<ExprAST> ParseMHSAExpr() {
  
  getNextToken(); // eat the MHSA.
  
  if (CurTok != '[')
    return LogError("MHSA declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "init_gpt";
  std::vector<int> notators_vec;
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else if (in_str(IdentifierStr, notators_str)){
        notators_vec.push_back(NotatorsMap[IdentifierStr]);
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  

  if (CurTok != ']')
    return LogError("Expected ] at MHSA.");
    getNextToken();

  if (dims.size()!=3)
    return LogError("MHSA requires input vocabulary size and output hiddens count.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected MHSA identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "MHSAForward";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers after MHSA.");
  }



  auto aux = std::make_unique<MHSAExprAST>(std::move(VarNames), "mhsa",
                                             std::move(dims[0]), std::move(dims[1]), std::move(dims[2]),
                                             std::move(notators_vec),
                                             init);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}






//
static std::unique_ptr<ExprAST> ParseMaxPool2dExpr() {
  std::string type;
  if (CurTok==tok_maxpool2d)
    type = "max";
  if (CurTok==tok_avgpool2d)
    type = "avg";

  getNextToken(); // eat the MaxPool2d.
  
  if (CurTok != '[')
    return LogError("MaxPool2d declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  

  if (CurTok != ']')
    return LogError("Expected ].");
    getNextToken();

  if (dims.size()<3)
    return LogError("MaxPool2d requires kernel size, stride and padding.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected conv identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "MaxPoolForward2d";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Esperado um ou mais identificadores após var.");
  }



  auto aux = std::make_unique<MaxPool2dExprAST>(std::move(VarNames), type,
                                             std::move(dims[0]), std::move(dims[1]), std::move(dims[2]));
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}





//
static std::unique_ptr<ExprAST> ParseBatchNorm2dExpr() {
  std::string type;

  getNextToken(); // eat the BatchNorm2d.
  
  if (CurTok != '[')
    return LogError("BatchNorm2d declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  

  if (CurTok != ']')
    return LogError("Expected ].");
    getNextToken();

  if (dims.size()<1)
    return LogError("BatchNorm2d requires input channels, kernel size, stride and padding.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected BatchNorm2d identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "BatchNormForward2d";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers at BatchNorm2d.");
  }



  auto aux = std::make_unique<BatchNorm2dExprAST>(std::move(VarNames), type,
                                             std::move(dims[0]));
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}




//
static std::unique_ptr<ExprAST> ParseBN2dReluExpr() {
  std::string type;

  getNextToken(); // eat the BatchNorm2d.
  
  if (CurTok != '[')
    return LogError("BN2dRelu declaration expected [");
    getNextToken();

  std::vector<std::unique_ptr<ExprAST>> dims;
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::string init = "";
  //std::make_unique<NumberExprAST>(NumVal)
  
  while (true) {
    if (CurTok != tok_number && CurTok != tok_identifier && CurTok != tok_self)
      return LogError("Expected tensor dimension number.");
    
    if (CurTok==tok_number)
    {
      if (std::fmod(NumVal, 1.0) != 0)
        LogWarning("Tensor dimensions must be of type int. They are not supposed to be float.");
    
      dims.push_back(std::make_unique<NumberExprAST>( (float)((int)round(NumVal)) ));
      getNextToken();
    } else if (CurTok==tok_identifier)
      if (in_str(IdentifierStr, tensor_inits))
      {
        init = IdentifierStr;
        getNextToken();
      } else
        dims.push_back(std::move(ParseIdentifierExpr()));
    else {
      dims.push_back(std::move(ParseSelfExpr()));
    }

    
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.
  }

  

  if (CurTok != ']')
    return LogError("Expected ].");
    getNextToken();

  if (dims.size()<1)
    return LogError("BN2dRelu requires input channels, kernel size, stride and padding.");


  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected BatchNorm2d identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "BN2dReluForward";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers at BN2dRelu.");
  }



  auto aux = std::make_unique<BN2dReluExprAST>(std::move(VarNames), type,
                                             std::move(dims[0]));
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}



//
static std::unique_ptr<ExprAST> ParseReluExpr() {
  std::string type;

  getNextToken(); // eat the Relu.
  
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  

  std::string pre_dot="";
  bool is_self = false;
  bool is_attr = false;
  if (CurTok == tok_self)
  {
    is_self=true;
    getNextToken();
  }
  if (CurTok == tok_class_attr)
  {
    is_attr=true;
    pre_dot = IdentifierStr;
    std::cout << "Obj attr pinned_tensor: " << pre_dot << ".\n";
    getNextToken();
  }

  if (CurTok != tok_identifier)
    return LogError("Expected Relu identifier name.");



  while (true) {
    std::string Name = IdentifierStr;
    
    getNextToken(); // eat identifier.

    
    std::unique_ptr<ExprAST> Init = nullptr;
    VarNames.push_back(std::make_pair(Name, std::move(Init)));
    functionVars[Name] = "ReluForward";

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("Expected one or more identifiers at Relu.");
  }



  auto aux = std::make_unique<ReluExprAST>(std::move(VarNames), type);
  aux->SetSelf(is_self);
  aux->SetIsAttribute(is_attr);
  aux->SetPreDot(pre_dot);

  
  if (CurTok==tok_space)
    getNextToken();
  
  return aux;
}





static std::unique_ptr<ExprAST> ParseLockExpr(std::string class_name="") {
  int cur_level_tabs = SeenTabs;
  getNextToken(); // eat the lock.


  std::string Name = "mutex";
  if (CurTok==tok_str)
  {
    Name = IdentifierStr;
    getNextToken(); // eat lock string "" name
  }


  
  if (lockVars.count(Name) == 0)
  {
    pthread_mutex_t* _mutex = new pthread_mutex_t;
    if (pthread_mutex_init(_mutex, NULL) != 0) {
      printf("Mutex initialization failed\n");
      return nullptr;
    }
    
    lockVars[IdentifierStr] = _mutex;  
  }
  
  
  std::vector<std::unique_ptr<ExprAST>> Body = ParseIdentedBodies(cur_level_tabs, class_name);


  return std::make_unique<LockExprAST>(std::move(Body), Name);
}



static std::unique_ptr<ExprAST> ParseNoGradExpr(std::string class_name="") {
  int cur_level_tabs = SeenTabs;
  getNextToken(); // eat no_grad
  
  std::vector<std::unique_ptr<ExprAST>> Body = ParseIdentedBodies(cur_level_tabs, class_name);

  return std::make_unique<NoGradExprAST>(std::move(Body));
}



static std::unique_ptr<ExprAST> ParseMustBeVar(std::string class_name="", std::string expr_name="") {

  std::unique_ptr<ExprAST> expr;

  if (CurTok==tok_class_attr||CurTok==tok_self)
    expr = ParseSelfExpr(class_name);
  else if (CurTok==tok_identifier)
    expr = ParseIdentifierExpr(class_name);
  else
  {
    std::string _error = expr_name + " expression expected a simple identifier, not another expression.";
    LogError(_error);
  }

  return std::move(expr);
}



static std::unique_ptr<ExprAST> ParseGlobalExpr(std::string class_name="") {
  getNextToken(); // eat global
  std::cout << "Parsing global expr\n";


  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("Expected identifier after global.");

  while (true) {


    if (CurTok!=tok_identifier)
      return LogError("Global expression must contain identifiers only.");

    ParseIdentifierExpr(class_name);
    globalVars.push_back(IdentifierStr);

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    
  }

  if (CurTok==tok_space)
    getNextToken();


  return std::make_unique<NumberExprAST>(0.0f);
}


static std::unique_ptr<ExprAST> ParseReturnExpr(std::string class_name="") {
  getNextToken(); // eat return
  std::cout << "Parsing var expr\n";

  std::vector<std::unique_ptr<ExprAST>> Vars, Destiny;
  std::vector<bool> IsAs;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("Expected identifier after return.");

  while (true) {
    std::unique_ptr<ExprAST> expr, aux;


    expr = ParseMustBeVar(class_name, "return");

    
    if (CurTok == tok_as)
    {
      getNextToken(); // eat as
      aux = std::move(expr);

      expr = ParseMustBeVar(class_name, "return");

      IsAs.push_back(true);
      Vars.push_back(std::move(aux));
      Destiny.push_back(std::move(expr));

    } else {
      Destiny.push_back(std::move(expr));

      expr = std::make_unique<NumberExprAST>(0.0f);
      IsAs.push_back(false);
      Vars.push_back(std::move(expr));
    }

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    
  }

  if (CurTok==tok_space)
    getNextToken();


  return std::make_unique<ReturnExprAST>(std::move(Vars), std::move(IsAs), std::move(Destiny));
}




/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
///   ::= ifexpr
///   ::= forexpr
///   ::= varexpr
static std::unique_ptr<ExprAST> ParsePrimary(std::string class_name="") {
  switch (CurTok) {
  default:
    //return std::move(std::make_unique<NumberExprAST>(0.0f));
    return LogErrorT(CurTok);
  case tok_identifier:
    return ParseIdentifierExpr();
  case tok_class_attr:
    return ParseSelfExpr(class_name);
  case tok_self:
    return ParseSelfExpr(class_name);
  case tok_number:
    return ParseNumberExpr();
  case tok_str:
    return ParseStringExpr();
  case tok_var_str:
    return ParseStrExpr();
  case tok_str_vec:
    return ParseStrVecExpr();
  case tok_float_vec:
    return ParseStrVecExpr();
  case '(':
    return ParseParenExpr();
  case tok_if:
    return ParseIfExpr(class_name);
  case tok_for:
    return ParseForExpr(class_name);
  case tok_while:
    return ParseWhileExpr(class_name);
  case tok_async_finish:
    return ParseFinishExpr(class_name);
  case tok_async:
    return ParseAsyncExpr(class_name);
  case tok_lock:
    return ParseLockExpr(class_name);
  case tok_no_grad:
    return ParseNoGradExpr(class_name);
  case tok_return:
    return ParseReturnExpr(class_name);
  case tok_var:
    return ParseVarExpr(class_name);
  case tok_tensor:
    return ParseTensorExpr(class_name);
  case tok_param:
    return ParseTensorExpr(class_name);
  case tok_pinned_tensor:
    return ParsePinnedTensorExpr();
  case tok_conv2d:
    return ParseConv2dExpr();
  case tok_global:
    return ParseGlobalExpr();
  case tok_lstm:
    return ParseLSTMExpr();
  case tok_embedding:
    return ParseEmbeddingExpr();
  case tok_mhsa:
    return ParseMHSAExpr();
  case tok_linear:
    return ParseLinearExpr();
  case tok_maxpool2d:
    return ParseMaxPool2dExpr();
  case tok_avgpool2d:
    return ParseMaxPool2dExpr();
  case tok_batchnorm2d:
    return ParseBatchNorm2dExpr();
  case tok_bn2drelu:
    return ParseBN2dReluExpr();
  case tok_relu:
    return ParseReluExpr();
  case '[':
    return ParseNewVector(class_name);
  case tok_space:
    getNextToken();
    return ParsePrimary(class_name);
  }
}

/// unary
///   ::= primary
///   ::= '!' unary
static std::unique_ptr<ExprAST> ParseUnary(std::string class_name="") {
  //std::cout <<"Parse unary\n";
  if(CurTok==tok_space)
    getNextToken();
  // If the current token is not an operator, it must be a primary expr.
  
  //std::cout << "Unary current token " << ReverseToken(CurTok) << "\n";
  if (!isascii(CurTok) || CurTok == '(' || CurTok == ',' || CurTok == '[')
  {
    //std::cout << "Returning, non-ascii found.\n";
    return ParsePrimary(class_name);
  }
  
  
  // If this is a unary operator, read it.
  int Opc = CurTok;
  
  //std::cout << "Unary expr\n";
  getNextToken();
  if (auto Operand = ParseUnary(class_name))
    return std::make_unique<UnaryExprAST>(Opc, std::move(Operand));
  return nullptr;
}


/// binoprhs
///   ::= ('+' unary)*
static std::tuple<std::unique_ptr<ExprAST>, int> ParseBinOpRHS(int ExprPrec,
                                              std::unique_ptr<ExprAST> LHS,
                                              std::string class_name="") {
  
  
  int LhsTok = 0;
  int RhsTok = 0;

  int L_cuda = type_float;
  int R_cuda = type_float;

  std::string LName, RName;
  if (LHS->GetType()=="tensor")
    L_cuda = type_tensor;
  if (LHS->GetType()=="pinned_tensor")
    L_cuda = type_pinned_tensor;
  if (LHS->GetType()=="str")
    L_cuda = type_string;

  while (true)
  {
    // If this is a binop, find its precedence.
    int TokPrec = get_tokenPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    

    if (TokPrec==BinopPrecedence[':'])
    {
      getNextToken();
      return std::make_tuple(std::move(LHS), L_cuda);
    }
    if (TokPrec < ExprPrec)
      return std::make_tuple(std::move(LHS), L_cuda);
    
      


    if (CurTok == tok_space)
    {
      //std::cout << "Returning tok space with " << SeenTabs << " tabs. \n\n\n";
      getNextToken();
      return std::make_tuple(std::move(LHS), L_cuda);
    }

    int BinOp = CurTok;


    /*
    //todo: it somehow jumps wrong op placements
    std::cout << "\n\nCur tok: " << CurTok << "\n";
    std::cout << in_char(CurTok, ops) <<  "\n";

    if (not in_char(CurTok, ops))
    {
      LogErrorBreakLine("Operador desconhecido.");
      return std::make_tuple(nullptr,0);
    }
    
    std::cout << "Cur tok post error: " << CurTok << "\n";
    */



    if(CurTok==':')
    {
      getNextToken();
      return std::make_tuple(std::move(LHS),L_cuda);
    }

    if (CurTok==')')
      return std::make_tuple(std::move(LHS),L_cuda);

    
    getNextToken(); // eat binop
    if (CurTok==tok_number)
      RName = std::to_string(NumVal);
    else
      RName = IdentifierStr;


    // Get the Right Hand Side token for debugging only
    RhsTok = CurTok;

    
    auto RHS = ParseUnary(class_name); // Returns an identifier, number or expression result
    if (!RHS)
      return std::make_tuple(nullptr,0);


    if (RHS->GetType()=="tensor")
      R_cuda=type_tensor;
    if (RHS->GetType()=="pinned_tensor")
      R_cuda=type_pinned_tensor;
    if (RHS->GetType()=="object"||RHS->GetType()=="object_vec")
      R_cuda=type_object;
    if (RHS->GetType()=="str")
      R_cuda = type_string;
    
    
    

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    int NextPrec = get_tokenPrecedence();
    

    if (TokPrec < NextPrec)
    {
      //std::cout << NextPrec << " Next Prec\n";
        
      auto tuple = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
      RHS = std::move(std::get<0>(tuple));
      R_cuda = std::get<1>(tuple);


      if (!RHS)
        return std::make_tuple(nullptr,0);
      
    }


    //std::cout << "\nBinary expression of BinOp and Rhs:" << "\n";
    //std::cout << ReverseToken(BinOp) << " " << ReverseToken(RhsTok) << "\n";
    //std::cout << "L type: " << L_cuda << " R type: " << R_cuda << "\n\n";


    if (L_cuda==type_string||R_cuda==type_string)
    {
      LHS = std::make_unique<ConcatStringsExprAST>(BinOp, std::move(LHS), std::move(RHS));
      LHS->SetType("str");
    }
    else if (R_cuda==type_object)
    {
      LHS = std::make_unique<BinaryObjExprAST>(BinOp, std::move(LHS), std::move(RHS));
      LHS->SetType("object");
    }
    else if (L_cuda==type_tensor && R_cuda==type_pinned_tensor)
    {
      //std::cout << "\nParse BinaryTensorPinned " << ReverseToken(BinOp) <<  "\n";
      LHS = std::make_unique<BinaryTensorPinnedExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
      LHS->SetType("tensor");
    }
    else if (L_cuda==type_pinned_tensor && R_cuda==type_float)
    {
      LHS = std::make_unique<BinaryPinnedScalarExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
      LHS->SetType("pinned_tensor");
    }
    else if (L_cuda==type_pinned_tensor && R_cuda==type_tensor)
    {
      LHS = std::make_unique<BinaryPinnedAndTensorExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
      LHS->SetType("pinned_tensor");
    }
    else if (L_cuda==type_tensor && R_cuda==type_float)
    {
      LHS = std::make_unique<BinaryTensorScalarExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
      LHS->SetType("tensor");  
    }
    else if (L_cuda==type_float && R_cuda==type_tensor)
    {
      std::cout << "Reverse LHS and RHS\n";
      //std::cout << "Bin op: " << BinOp << "\n";


      if (BinOp=='/')
        BinOp = 77; // scalar / tensor

      if (BinOp=='-') // inversion of 1 - tensor
      {
        RHS = std::make_unique<BinaryTensorScalarExprAST>('*',
                                                    std::move(RHS),
                                                    std::move(std::make_unique<NumberExprAST>(-1.0f)));
                                                    //std::move(LHS)
                                                    
        LHS = std::make_unique<BinaryTensorScalarExprAST>('+',
                                                    std::move(RHS), std::move(LHS));
      } else {
        if (BinOp!=':') // Avoid codegen reversing //todo: is this necessary anymore?
          LHS = std::make_unique<BinaryTensorScalarExprAST>(BinOp,
                                                    std::move(RHS), std::move(LHS));
        else
          LHS = std::make_unique<BinaryTensorScalarExprAST>(BinOp,
                                                    std::move(LHS), std::move(RHS));
      }
      LHS->SetType("tensor");  
      L_cuda=type_tensor;
      R_cuda=type_float;
    }
    else if (L_cuda==type_tensor && R_cuda==type_tensor)
    { 
      LHS = std::make_unique<BinaryTensorTensorExprAST>(BinOp,
                                                      std::move(LHS), std::move(RHS));
      R_cuda=type_float;
      LHS->SetType("tensor");
    }
    else
      LHS = std::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
     

    LhsTok = RhsTok;    
  
  }
}


/// expression
///   ::= unary binoprhs
///
static std::unique_ptr<ExprAST> ParseExpression(std::string class_name) {
  
  //std::cout << "Parse Expression\n";
  
  auto LHS = ParseUnary(class_name);
  if (!LHS)
    return nullptr;

  return std::get<0>(ParseBinOpRHS(0, std::move(LHS), class_name));
}

/// prototype
///   ::= id '(' id* ')'
///   ::= binary LETTER number? (id, id)
///   ::= unary LETTER (id)
static std::unique_ptr<PrototypeAST> ParsePrototype(std::string class_name="") {
  std::string FnName = class_name;
  std::string _class, method;
  method = "";
  _class = class_name;

  unsigned Kind = 0; // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok) {
  default:
    return LogErrorP("Esperado nome da função no protótipo");
  case tok_identifier:
    FnName += IdentifierStr;
    method = IdentifierStr;
    Kind = 0;
    getNextToken();
    break;
  case tok_unary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP("Esperado operador unário");
    FnName += "unary";
    FnName += (char)CurTok;
    Kind = 1;
    getNextToken();
    break;
  case tok_binary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP("Esperado operador binário");
    FnName += "binary";
    FnName += (char)CurTok;
    Kind = 2;
    getNextToken();

    // Read the precedence if present.
    if (CurTok == tok_number) {
      if (NumVal < 1 || NumVal > 100)
        return LogErrorP("Precedência inválida: deve ser entre 1 e 100");
      BinaryPrecedence = (unsigned)NumVal;
      getNextToken();
    }
    break;
  }

  if (CurTok != '(')
    return LogErrorP("Esperado '(' no protótipo");


  getNextToken();


  std::string type;
  std::vector<std::string> ArgNames, Types;


  if (class_name!="") // If it is a class method, add self
  {
    Types.push_back("s");
    ArgNames.push_back("self");
  }
  Types.push_back("s");
  ArgNames.push_back("scope_str");
  Types.push_back("s");
  ArgNames.push_back("previous_scope");
  Types.push_back("i");
  ArgNames.push_back("thread_id");
  Types.push_back("i");
  ArgNames.push_back("has_grad");

  while (CurTok != ')')
  {
    type="str";
    if (IdentifierStr=="t")
      type="tensor";
    if (IdentifierStr=="c")
      type="function";
    if (IdentifierStr=="f")
      type="float";

    if (IdentifierStr!="t" && IdentifierStr!="f" && IdentifierStr!="s" && IdentifierStr!="c")
      LogErrorP_to_comma("Prototype var type must be t, f, s or c");
    else {
      Types.push_back(IdentifierStr);
      getNextToken(); // eat arg type

      ArgNames.push_back(IdentifierStr);

      if (type=="float")
        floatVars.push_back(IdentifierStr);
      else if (type=="tensor")
        tensorVars.push_back(IdentifierStr);
      else if (type=="function")
        functionVars[IdentifierStr] = "ConvForward2d";
      else
        strVars.push_back(IdentifierStr);
      
      getNextToken(); // eat arg name
    }
    


    if (CurTok == ')')
        break;
      
    if (CurTok != ',')
    {
      return LogErrorP("Expected ')' or ',' at prototype arguments list.");
    }
    getNextToken();
  }

  // success.
  getNextToken(); // eat ')'.

  // Verify right number of names for operator.
  if (Kind && ArgNames.size() != Kind)
    return LogErrorP("Número inválido de operandos para o operador");

  if (CurTok!=tok_space)
    LogError("Post prototype parsing requires a line break.");
  getNextToken();


  return std::make_unique<PrototypeAST>(FnName, _class, method, ArgNames, Types, Kind != 0,
                                         BinaryPrecedence);
}



/// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition(std::string class_name="") {

  int cur_level_tabs = SeenTabs;

  getNextToken(); // eat def.


  auto Proto = ParsePrototype(class_name);
  if (!Proto)
  {
    std::string _error = "Error defining " + class_name + " prototype.";  
    LogError(_error);
    return nullptr;
  } 
  
  
  std::vector<std::unique_ptr<ExprAST>> Body;

  while(!in_char(CurTok, terminal_tokens))
  {
    if (SeenTabs <= cur_level_tabs && CurTok != tok_space)
      break;
      

    if (CurTok==tok_space)
      getNextToken();

    if (SeenTabs <= cur_level_tabs)
      break;

    Body.push_back(std::move(ParseExpression(class_name)));
  }

  //std::cout << "function number of expressions: " << Body.size() << "\n";

  if (Body.size()==0)
  {
    std::string _error = "Function " + class_name + "'s body was not declared.";  
    LogError(_error);
    return nullptr;
  } 

  return std::make_unique<FunctionAST>(std::move(Proto), std::move(Body));
  
}


/// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
  //std::cout << "Top Level Expression\n";

  
  std::vector<std::unique_ptr<ExprAST>> Body;
  while(!in_char(CurTok, terminal_tokens))
  {
    Body.push_back(std::move(ParseExpression()));
    //std::cout << "\n\nTop level expr cur tok: " << ReverseToken(CurTok) <<  ".\n";
    //std::cout << "Top level expr number of expressions: " << Body.size() <<  ".\n\n\n";
  }
  

  // Make an anonymous proto.
  auto Proto = std::make_unique<PrototypeAST>("__anon_expr", "", "",
                                                std::vector<std::string>(),
                                                std::vector<std::string>());
    
  return std::make_unique<FunctionAST>(std::move(Proto), std::move(Body));
  
  return nullptr;
}



/// external ::= 'extern' prototype
static std::unique_ptr<PrototypeAST> ParseExtern() {
  getNextToken(); // eat extern.
  return ParsePrototype();
}






//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//

//global
static std::unique_ptr<KaleidoscopeJIT> TheJIT;
static std::unique_ptr<LLVMContext> TheContext;
static std::unique_ptr<LLVMContext> GlobalContext = std::make_unique<LLVMContext>();


static std::unique_ptr<IRBuilder<>> Builder;
static std::unique_ptr<Module> TheModule;
static std::unique_ptr<Module> GlobalModule;


static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
static ExitOnError ExitOnErr;


// Vars
static std::map<std::string, Value *> NamedValues;
static std::map<std::string, char *> NamedStrs;
static std::map<std::string, AllocaInst *> NamedStrVecs;
static std::map<std::string, AllocaInst *> NamedFloatVecs;
static std::map<std::string, std::vector<char *>> ClassStrVecs;
static std::map<std::string, std::vector<float>> ClassFloatVecs;
static std::map<std::string, float> NamedClassValues;
static std::map<std::string, std::string> NamedObjects;

static std::map<std::string, std::vector<std::pair<std::string, std::string>>> ScopeVarsToClean;
static std::map<std::string, char *> ScopeNamesToClean;
static std::map<int, std::map<std::string, std::vector<std::string>>> ThreadedScopeTensorsToClean;


// Aux to not lose pointers
std::map<std::string, std::string> AuxRandomStrs;
std::map<std::string, std::vector<char *>> StrVecAuxHash;
std::map<std::string, std::vector<float>>  FloatVecAuxHash;



// Optimizer
static std::map<std::string, float *> NamedParamGrads;


// File Handling
std::vector<char *> glob_str_files;




// Handle Class self with phantom argument



Value * VoidPtr_toValue(void *vec)
{
  auto void_ptr_ty = Type::getInt8Ty(*TheContext)->getPointerTo();
  Value* LLVMValue = ConstantInt::get(Type::getInt64Ty(*TheContext), reinterpret_cast<uint64_t>(vec));
  return Builder->CreateIntToPtr(LLVMValue, void_ptr_ty);
}

Value* FloatPtr_toValue(float* vec)
{
    // Get the type for float*
    auto float_ptr_ty = Type::getFloatTy(*TheContext)->getPointerTo();
    
    // Convert the float* to uint64_t and create a constant integer value
    Value* LLVMValue = ConstantInt::get(Type::getInt64Ty(*TheContext), reinterpret_cast<uint64_t>(vec));
    
    // Cast the integer value to float*
    return Builder->CreateIntToPtr(LLVMValue, float_ptr_ty);
}



static std::unique_ptr<ExprAST> ParseClass() {
  getNextToken(); // eat class.

  if (CurTok != tok_identifier)
    return LogError("Expected class name");
  std::string Name = IdentifierStr;

  Classes.push_back(Name);

  getNextToken();

  if(CurTok==tok_space)
    getNextToken();
  

  if (CurTok!=tok_def)
    return LogError("A class definition requires it's functions.");

  int i=0;
  while(CurTok==tok_def)
  {
    
    auto Func = ParseDefinition(Name);
    if (!Func)
      return nullptr;
      //return LogError("Falha no parsing da função da Classe.");
    if (!ends_with(Func->getProto().getName(),"__init__") && i==0)
      return LogError("Class requires __init__ method");
    
    //std::cout << "THE FUNCTION WAS CREATED AS: " << Func->getProto().getName() << "\n";


    std::string proto_name = Func->getProto().getName();    


    FunctionProtos[proto_name] =
      std::make_unique<PrototypeAST>(Func->getProto());
    ExitOnErr(TheJIT->addAST(std::move(Func)));

    if(CurTok==';')
      getNextToken();

    if(CurTok==tok_space)
      getNextToken();

    i+=1;
  }
  
  return nullptr;
}



extern "C" void sleep(float id)
{
  std::cout << "\n\nSleep " << id << " begin" << "\n";
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> dis(3, 7); // Generate between 1 and 100
  int random_number = dis(gen);

  //std::this_thread::sleep_for(std::chrono::seconds(random_number));
  std::this_thread::sleep_for(std::chrono::seconds((int)id));

  std::cout << "Sleep " << id << " finish" << "\n";

  //return id;
}


extern "C" float silent_sleep(float id)
{
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> dis(3, 7); // Generate between 1 and 100
  int random_number = dis(gen);

  //std::this_thread::sleep_for(std::chrono::seconds(random_number));
  std::this_thread::sleep_for(std::chrono::seconds((int)id));

 
  return 0;
}


std::chrono::high_resolution_clock::time_point START_TIME;

extern "C" float start_timer(float id)
{
  START_TIME = std::chrono::high_resolution_clock::now();
 
  return 0;
}

extern "C" float end_timer(float id)
{
  std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsedTime = endTime - START_TIME;

  // Print the elapsed time in seconds
  std::cout << "Elapsed time: " << elapsedTime.count() << " seconds.\n";

  //std::cout << "Length " << rds.size() << "\n";

  return 0;
}



std::vector<float> format_BatchFirst_Dims(std::vector<float> dims)
{
  std::vector<float> new_dims;
  new_dims.push_back(dims[0]);
  int aux=1;
  for (int i = 0; i < dims.size()-1; i++)
    aux *= dims[i+1];
  new_dims.push_back(aux);
  return new_dims;
}


std::vector<float> format_LinearLayer_Dims(std::vector<float> dims)
{
  std::vector<float> new_dims;
  int aux=1;
  for (int i = 0; i < dims.size()-1; i++)
    aux *= dims[i];
  new_dims.push_back(aux);
  new_dims.push_back(dims[dims.size()-1]);
  return new_dims;
}



int resultingDimsProdOnMult(std::vector<float> Ldims, std::vector<float> Rdims)
{
  float aux=1;
  for (int i = 0; i < Ldims.size()-1; i++)
    aux = aux * Ldims[i];
  aux = aux * Rdims[0];
  return (int)aux;
}


float *get_from_pool(int, float, std::string);

extern "C" void *gpu(int thread_id, Tensor *tensor, Tensor *pinned_tensor)
{
  //std::cout << "\nGpu transfer for: " << tensor.name << " on worker " << idx << "\n";
  
  float *tensor_ptr, *tensor_cpu;

  
  tensor_cpu = pinned_tensor->cpu_tensor_ptr;
  std::vector<float> dims = pinned_tensor->dims;
  float dims_prod = pinned_tensor->dims_prod;
  



  
  if (tensor->dims_prod==dims_prod)
    tensor_ptr = tensor->tensor_ptr;
  else
    tensor_ptr = get_from_pool(thread_id, dims_prod, "gpu");
  
  //tensor_ptr = get_from_pool(dims_prod, "gpu");


  
  Loader *loader=nullptr;
  CudaStreams *cuda_stream=nullptr;
  
  
  cuda_stream = AllocateStream();
  cudaMemcpyAsync(tensor_ptr, tensor_cpu, dims_prod * sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
  //cudaMemcpy(tensor_ptr, tensor_cpu, dims_prod * sizeof(float), cudaMemcpyHostToDevice);
  pinned_tensor->cuda_stream = cuda_stream;
  



  if (nn_mode==eval_mode)
  {

  } else {
    
    Tensor *attr_tensor;
    attr_tensor = createTensor(tensor_ptr, dims, dims_prod, true, "");
    attr_tensor->op = gpu_op;
    todo_backward_tensors.push_back(attr_tensor); // pass to gc
    
  }

  tensor->AttrTensor(tensor_ptr, dims, dims_prod, cuda_stream, loader);
  tensor->from_grad_or_load = true;

  return 0;
}



extern "C" float gpuw(int thread_id, Tensor *tensor, Tensor *pinned_tensor, float idx)
{
  //std::cout << "\nGpu transfer for: " << tensor->name << " on worker " << idx << "\n";
  
  float *tensor_ptr, *tensor_cpu;

  
  
  std::vector<float> dims, batchless_dims;
  dims = pinned_tensor->dims;
  

  batchless_dims = BatchLessDims(dims);
  float batchless_dims_prod = (float)DimsProd(batchless_dims);


  tensor_cpu = pinned_tensor->cpu_tensor_ptr + static_cast<int>(idx*batchless_dims_prod);

  
  if (tensor->dims_prod==batchless_dims_prod)
    tensor_ptr = tensor->tensor_ptr;
  else
    tensor_ptr = get_from_pool(thread_id, batchless_dims_prod, "gpuw");
  
  //tensor_ptr = get_from_pool(batchless_dims_prod, "gpuw");


  
  Loader *loader=nullptr;
  CudaStreams *cuda_stream=nullptr;
  
  
  if (batchless_dims_prod<2000){
    cudaMemcpy(tensor_ptr, tensor_cpu, batchless_dims_prod * sizeof(float), cudaMemcpyHostToDevice);
  }
  else// if (batchless_dims_prod<1000)
  {
    cuda_stream = AllocateStream();
    cudaMemcpyAsync(tensor_ptr, tensor_cpu, batchless_dims_prod * sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
    //cudaMemcpy(tensor_ptr, tensor_cpu, batchless_dims_prod * sizeof(float), cudaMemcpyHostToDevice);
    pinned_tensor->cuda_stream = cuda_stream;
  }
  /*
  else
  {
    //cuda_stream = AllocateStream();
    //cudaMemcpyAsync(tensor_ptr, tensor_cpu, batchless_dims_prod * sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
    loader = new Loader();
    loader->Load(tensor_ptr, tensor_cpu, batchless_dims_prod);
  }
  */



  if (nn_mode==eval_mode)
  {

  } else {
    
    Tensor *attr_tensor;
    attr_tensor = createTensor(tensor_ptr, batchless_dims, batchless_dims_prod, true, "");
    attr_tensor->op = gpu_op;
    todo_backward_tensors.push_back(attr_tensor); // pass to gc
    
  }

  tensor->AttrTensor(tensor_ptr, batchless_dims, batchless_dims_prod, cuda_stream, loader);
  tensor->from_grad_or_load = true;

  return 0;
}



std::vector<float> NewDimsOnMult(std::vector<float> Ldims, std::vector<float> Rdims)
{
  

  std::vector<float> new_dims;
  if (Ldims[Ldims.size()-1]!=Rdims[Rdims.size()-1])
  {
    LogError("The last dimension of multiplied tensors must be the same.");
    std::cout << "Dim LHS: ";
    PrintDims(Ldims);
    std::cout << "Dim RHS: ";
    PrintDims(Rdims);
    return {}; 
  }
  for (int i = 0; i < Ldims.size()-1; i++)
    new_dims.push_back(Ldims[i]);
  new_dims.push_back(Rdims[0]);


  return new_dims;
}



extern "C" void *NewDimsOnIdx(std::vector<float> dims)
{
  std::vector<float> new_dims;

  for (int i = 0; i < dims.size()-1; i++)
    new_dims.push_back(dims[i+1]);


  // Aux to not lose pointers
  std::string random_str = RandomString(15);
  NamedDims[random_str] = new_dims; // Deal with new_dims being deleted after scope finished.
  AuxRandomStrs[random_str] = "dim";


  std::cout << "NewDimsOnIdx" << "\n";
  PrintDims(NamedDims[random_str]);

  return &NamedDims[random_str];
}


extern "C" float Add(float value, float v2)
{
  return value + v2; 
}


extern "C" void PrintFloat(float value){
  std::cout << "Printing float.\n";
  std::cout << "Float value: " << value << "\n";
}

extern "C" float unbug(){
  return 0;
}

extern "C" float UnbugFloat(float value){
  return value;
}


extern "C" float PrintStr(char* value){
  std::cout << "Str: " << value << "\n";
  return 0;
}


extern "C" float PrintStrVec(std::vector<char*> vec)
{
  for (int i=0; i<vec.size(); i++)
    std::cout << vec[i] << "\n";

  return 0;
}


extern "C" float PrintFloatVec(std::vector<float> vec)
{

  std::cout << "Float vector:\n[";
  for (int i=0; i<vec.size()-1; i++)
    std::cout << "" << vec[i] << ", ";
  std::cout << "" << vec[vec.size()-1];
  std::cout << "]\n\n";

  return 0;
}


extern "C" float first_nonzero(char *self)
{
  //std::cout << "first_nonzero call of: " << self <<"\n";

  std::vector<float> vec;
  vec = ClassFloatVecs[self];
  
  /*
  std::cout << "[";
  for (int i=0; i<vec.size(); i++)
    std::cout << vec[i] << ", ";
  std::cout << "]" << "\n";
  */


  float idx = -1;
  for (int i=0; i<vec.size(); i++)
    if (vec[i]!=0)
    {
      idx = i;
      break;
    }

  delete[] self;
  return idx;
}



extern "C" float LenStrVec(std::vector<char*> vec)
{
  return (float) vec.size();
}



extern "C" void * ShuffleStrVec(std::vector<char*> vec)
{
  std::random_device rd;
  std::mt19937 g(rd()^get_millisecond_time());


  std::shuffle(vec.begin(), vec.end(), g);

  
  return &vec;
}


//deprecated
extern "C" char * shuffle_str(char *string_list)
{

  std::ostringstream oss;

  std::vector<std::string> splitted = split(string_list, "|||");


  std::random_shuffle(splitted.begin(), splitted.end());

  for (int i=0; i<splitted.size(); i++)
  {
    if (i>0)
      oss << "|||";
    oss << splitted[i];
  }

  std::string result = oss.str();

  char * cstr = new char [result.length()+1];
  std::strcpy (cstr, result.c_str());
    
  return cstr;
}




//std::map<float, std::vector<float *>> TensorPool;
std::map<int, std::map<float, std::vector<float *>>> TensorPool;

void move_to_pool(int thread_id, float dims_prod, float *tensor_ptr, std::string from)
{
  //if (dims_prod==50*256)
  //  std::cout << "push B*OC of " << from << "\n";

  if (dims_prod==0)
    return;
  //std::cout << "move_to_pool from: " << from << "\n";
  

  std::vector<float *> tensors_in_pool = TensorPool[thread_id][dims_prod];
  if (!in_float_ptr_vec(tensor_ptr, tensors_in_pool))
  {
    //if(!(tensors_in_pool.size()<30&&dims_prod==1))
    /*
    if(tensors_in_pool.size()<1000)
      TensorPool[dims_prod].push_back(tensor_ptr);
    else
    {
      std::cout << "FREEING TENSOR WITH dims prod: " << dims_prod << " from: " << from <<  "\n";
      cudaCheck(cudaFree(tensor_ptr));
    }
    */
    TensorPool[thread_id][dims_prod].push_back(tensor_ptr);
  }  
}

float *get_from_pool(int thread_id, float dims_prod, std::string from)
{
  //if (dims_prod==32)
  //  std::cout << "get B*OC of " << from << "\n";

  if (dims_prod==0)
    return nullptr;


  float *tensor_ptr;

  if(TensorPool[thread_id].count(dims_prod)>0)
  {
    std::vector<float *> tensors_in_pool = TensorPool[thread_id][dims_prod];
    if (tensors_in_pool.size()>0)
    {
      //std::cout << "GETTING FROM POOL: " << dims_prod << "\n";
      tensor_ptr = tensors_in_pool.back();
      TensorPool[thread_id][dims_prod].pop_back();
      return tensor_ptr;
    }
  }

  

  std::cout << "Malloc new space from " << from << " of size: " << dims_prod << ", at thread: " << thread_id << "\n";

  cudaCheck(cudaMalloc(&tensor_ptr, dims_prod*sizeof(float)));
  return tensor_ptr;
}




void move_to_pool_pow2(int thread_id, float dims_prod, float *tensor_ptr, std::string from)
{
  
  if (dims_prod==0)
    return;

  float nearest_ceil_pow2 = 1;
  while(nearest_ceil_pow2<dims_prod)
    nearest_ceil_pow2*=2;
  dims_prod = nearest_ceil_pow2;

  std::vector<float *> tensors_in_pool = TensorPool[thread_id][dims_prod];

  if (!in_float_ptr_vec(tensor_ptr, tensors_in_pool))
    TensorPool[thread_id][dims_prod].push_back(tensor_ptr);
  
}

float *get_from_pool_pow2(int thread_id, float dims_prod, std::string from)
{
  if (dims_prod==0)
    return nullptr;

  float nearest_ceil_pow2 = 1;
  while(nearest_ceil_pow2<dims_prod)
    nearest_ceil_pow2*=2;
  dims_prod = nearest_ceil_pow2;

  float *tensor_ptr;

  if(TensorPool[thread_id].count(dims_prod)>0)
  {
    std::vector<float *> tensors_in_pool = TensorPool[thread_id][dims_prod];
    if (tensors_in_pool.size()>0)
    {
      tensor_ptr = tensors_in_pool.back();
      TensorPool[thread_id][dims_prod].pop_back();
      return tensor_ptr;
    }
  }


  std::cout << "Malloc new space from " << from << " of size: " << dims_prod << ", at thread: " << thread_id << " with the nearest pow of 2\n";

  cudaCheck(cudaMalloc(&tensor_ptr, dims_prod*sizeof(float)));
  return tensor_ptr;
}










extern "C" void *LoadTensor(char *tensor_name){
  //std::cout << "\n\nLOAD TENSOR: " << tensor_name <<  "\n";
  Tensor *ret = NamedTensorsT[tensor_name];
  move_to_char_pool(strlen(tensor_name)+1, tensor_name, "free");
  //std::cout << "return load." << "\n";
  //delete[] tensor_name;
  return ret;
}

/*
extern "C" void *LoadDims(char *tensor_name) // TODO: invert this back
{
  std::cout << "LOADING DIMS"  << "\n";
  PrintDims(NamedDims[tensor_name]);

  std::string random_str = RandomString(15);
  NamedDims[random_str] = NamedDims[tensor_name];
  AuxRandomStrs[random_str] = "dim";
  delete[] tensor_name;

  return &NamedDims[random_str];
}
*/


extern "C" float print(char* str, float x){
  std::string _str = str;
  std::cout << "\n" << _str << " " << x << "\n";
  return 0;
}


extern "C" float PrintTensor(int thread_id, char* tensorName){
  std::cout << "Printing Tensor " << tensorName << " at stream " << thread_id << "\n";



  Tensor *tensor = NamedTensorsT[tensorName];
  int arr_size = tensor->dims_prod;
  float *tensor_cpu = new float[arr_size];

  
  std::vector<float> dims = tensor->dims;
  
  
  cudaStream_t stream = ThreadsStream[thread_id];
  tensor->Sync();
  cudaStreamSynchronize(stream);
  cudaDeviceSynchronize();
  cudaCheck(cudaMemcpy(tensor_cpu, tensor->tensor_ptr, arr_size*sizeof(float), cudaMemcpyDeviceToHost));


  std::cout << "\nTensor \033[95m" << tensorName << "\033[0m:\n\n";
  PrintDims(dims);
  std::cout << "\n";
  std::vector<float> ends;


  for (int i = 0; i < dims.size(); i++) {
    int prod=1;
    for (int j = 0; j <= i; j++)
      prod = prod*dims[dims.size()-1-j];
    ends.push_back(prod);
  }


  int line = 1;
  bool line_changed = true;

  
  //if (arr_size>2000)
  //  arr_size = 2000;

  for (int i = 0; i < arr_size; i++) {

    int to_prints = 0;

    for (int e = 0; e < ends.size(); e++)
    {
      if (fmod((arr_size-i),(int)ends[e]) == 0.0f)
        to_prints+=1;
    }

    if(to_prints>0)
    {
      for (int j=0; j<(dims.size()-to_prints); j++)
        std::cout << " ";
        
      for (int j=0; j<to_prints; j++)
        std::cout << "[";
    }
    

    //std::cout << "LAST SIZE " << dims[dims.size()-1] << " Mod: " << fmod(i, 1+dims[dims.size()-1]) << "\n";
    int precision;
    if (tensor_cpu[i]>=0)
      precision=4;
    else
      precision=3;
    std::cout << std::fixed  << std::setprecision(precision) << tensor_cpu[i];


    for (int e = 0; e < ends.size(); e++)
      if (fmod((i+1),(int)ends[e]) == 0.0f)
        std::cout << "],";
    

    if (i!=(arr_size-1))
    {
      if (fmod(i+1, dims[dims.size()-1]) == 0.0f)
      {
        line+=1;
        line_changed=true;
        std::cout << "\n";
      }
      else
        std::cout << ",  ";
    }

    if(fmod(i+1, ends[1]) == 0.0f)
      std::cout << "\n";


  }
  
  std::cout << "\n";
  PrintDims(dims);
  std::cout << "\n\n";

  delete[] tensor_cpu;

  return 0;
}



extern "C" float print_tensor(Tensor tensor){
  char* tensorName = new char[tensor.name.size() + 1]; // Allocate memory for the C-style string
  std::strcpy(tensorName, tensor.name.c_str()); // Copy the string

  PrintTensor(0, tensorName);

  delete[] tensorName;
  return 0;
}


extern "C" float PrintTensorF(const float *cuda_tensor, int d1, int d2){

  std::vector<float> dims;
  dims.push_back(d1);
  dims.push_back(d2);

  int arr_size = DimsProd(dims);


  float *tensor = new float[arr_size];
  //std::cout << "Printing Tensor " << arr_size << "\n";
  
  cudaDeviceSynchronize();
  cudaCheck(cudaMemcpy(tensor, cuda_tensor, arr_size*sizeof(float), cudaMemcpyDeviceToHost));


  
  std::cout << "\n";
  PrintDims(dims);
  std::vector<float> ends;


  for (int i = 0; i < dims.size(); i++) {
    int prod=1;
    for (int j = 0; j <= i; j++)
      prod = prod*dims[dims.size()-1-j];
    ends.push_back(prod);
  }


  int line = 1;
  bool line_changed = true;
  for (int i = 0; i < arr_size; i++) {

    int to_prints = 0;

    for (int e = 0; e < ends.size(); e++)
    {
      if (fmod((arr_size-i),(int)ends[e]) == 0.0f)
        to_prints+=1;
    }

    if(to_prints>0)
    {
      for (int j=0; j<(dims.size()-to_prints); j++)
        std::cout << " ";
        
      for (int j=0; j<to_prints; j++)
        std::cout << "[";
    }
    

    //std::cout << "LAST SIZE " << dims[dims.size()-1] << " Mod: " << fmod(i, 1+dims[dims.size()-1]) << "\n";
    int precision;
    if (tensor[i]>=0)
      precision=4;
    else
      precision=3;
    std::cout << std::fixed  << std::setprecision(precision) << tensor[i];


    for (int e = 0; e < ends.size(); e++)
      if (fmod((i+1),(int)ends[e]) == 0.0f)
        std::cout << "],";
    

    if (i!=(arr_size-1))
    {
      if (fmod(i+1, dims[dims.size()-1]) == 0.0f)
      {
        line+=1;
        line_changed=true;
        std::cout << "\n";
      }
      else
        std::cout << ",  ";
    }

    if(fmod(i+1, ends[1]) == 0.0f)
      std::cout << "\n";


  }
  std::cout << "\n";
  
  delete[] tensor;

  return 0;
}




Function *getFunction(std::string Name) {
  // First, see if the function has already been added to the current module.
  if (auto *F = TheModule->getFunction(Name))
    return F;

  // If not, check whether we can codegen the declaration from some existing
  // prototype.
  auto FI = FunctionProtos.find(Name);
  if (FI != FunctionProtos.end())
    return FI->second->codegen();

  // If no existing prototype exists, return null.
  return nullptr;
}

/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
static AllocaInst *CreateEntryBlockAlloca(Function *TheFunction,
                                          StringRef VarName) {
  IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                   TheFunction->getEntryBlock().begin());
  return TmpB.CreateAlloca(Type::getFloatTy(*TheContext), nullptr, VarName);
}



Value *NameSolverAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  //std::cout << "\n\n\nName solver type: " << Type << "\n\n\n\n";


  Value *name;
  int type;
  std::vector<std::unique_ptr<ExprAST>> idx;

  
  bool include_scope = GetSolverIncludeScope();
  Value *var_name = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});

  


  if(Names.size()>1)
    for (int i=0; i<Names.size()-1;i++)
    {
      name = Builder->CreateGlobalString(std::get<0>(Names[i]));
      type = std::get<1>(Names[i]);
      idx = std::move(std::get<2>(Names[i]));

      //std::cout << "NameSolver[" << i<< "]:  " << std::get<0>(Names[i]) << ", type: " << type << "\n";

      if (i==0)
      {
        if (type==type_self)
          var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeLeft"),
                                                          {var_name, first_arg});
        else
        {
          if((Type=="object"||Type=="tensor"||Type=="float"||type==type_object_name||Type=="str")&&include_scope)
            var_name = Builder->CreateCall(TheModule->getFunction("ConcatScopeStr"),
                                                          {var_name, scope_str});
        }
      }

      if (type==type_object_name)
      {
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeLeft"),
                                                        {var_name, name});
        var_name = Builder->CreateCall(TheModule->getFunction("LoadObject"),
                                                        {var_name});

      }

      if (type==type_attr||type==type_var)
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeLeft"),
                                                        {var_name, name});

      if (type==type_vec)
      {
        Value *_idx = idx[0]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeLeft"),
                                                        {var_name, name});
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatNumToStrFree"),
                                                        {var_name, _idx});
        var_name = Builder->CreateCall(TheModule->getFunction("LoadObjectScopeName"),
                                                        {var_name});
      }
    }


  if(NameSolveToLast)
  {
    if(Names.size()==1)// Concat scope only
      if((Type=="object"||Type=="tensor"||Type=="float"||Type=="str")&&include_scope)
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatScopeStr"),
                                                          {var_name, scope_str});



    name = Builder->CreateGlobalString(std::get<0>(Names[Names.size()-1]));
    idx = std::move(std::get<2>(Names[Names.size()-1]));

    //std::cout << "\n\n\nNAMESOLVER TYPE OF LAST: " << type << "\n";
    //std::cout << "For: " << std::get<0>(Names[Names.size()-1]) << "\n";
    //std::cout << "Type: " << Type << "\n";
    var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeLeft"),
                                                      {var_name, name});
    if (Type=="object_vec")
    {
      Value *_idx = idx[0]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatNumToStrFree"),
                                                        {var_name, _idx});
    }
  }

  return var_name;
}


Value *NumberExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  
  return ConstantFP::get(*TheContext, APFloat(Val));
}

Value *StringExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  SetName(Val);
  return Builder->CreateGlobalString(Val);
}




//===----------------------------------------------------------------------===//
// Dataset
//===----------------------------------------------------------------------===//


extern "C" void * zeros_vec(float size) {
  // TODO: turn into python like expression [0]*size

  std::vector<float> vec = std::vector<float>(static_cast<size_t>(size), 0.0f);
  

  // Aux to not lose pointers
  std::string random_str = RandomString(15);
  FloatVecAuxHash[random_str] = vec;
  AuxRandomStrs[random_str] = "float_vec";
    
  return &FloatVecAuxHash[random_str];
}

extern "C" void * ones_vec(float size) {
  // TODO: turn into python like expression [0]*size

  std::vector<float> vec = std::vector<float>(static_cast<size_t>(size), 1.0f);
  

  // Aux to not lose pointers
  std::string random_str = RandomString(15);
  FloatVecAuxHash[random_str] = vec;
  AuxRandomStrs[random_str] = "float_vec";
    
  return &FloatVecAuxHash[random_str];
}


extern "C" void * _glob_b_(char *pattern) {
  glob_t glob_result;

  std::vector<char *> ret;

  if (glob(pattern, GLOB_TILDE, NULL, &glob_result) == 0) {
      for (size_t i = 0; i < glob_result.gl_pathc; ++i) {

        ret.push_back(strdup(glob_result.gl_pathv[i]));
      }
      globfree(&glob_result);
  }


  if (ret.size()<1)
    LogErrorS("Glob failed to find files.");
    
  // Aux to not lose pointers
  std::string random_str = RandomString(15);
  StrVecAuxHash[random_str] = ret;
  AuxRandomStrs[random_str] = "str_vec";
    
  return &StrVecAuxHash[random_str];
}



float *current_data;



extern "C" float load_preprocess_img(Tensor tensor, char *img_name)
{
  float *img;
  img = load_img(img_name); 
  
  std::vector<float> dims = tensor.dims;

  
  // Last three dims.
  int img_dims_prod = dims[dims.size()-1]*dims[dims.size()-2]*dims[dims.size()-3];


  current_data = new float[img_dims_prod];
  cudaCheck(cudaMallocHost(&current_data, img_dims_prod*sizeof(float)));


  for (int j = 0; j < img_dims_prod; ++j)
    current_data[j] = img[j];
  delete[] img;


  float *x = tensor.tensor_ptr;
  //cudaMalloc(&x, img_dims_prod*sizeof(float));
  cudaCheck(cudaMemcpy(x, current_data, img_dims_prod*sizeof(float), cudaMemcpyHostToDevice));


  return 0;
}


//===----------------------------------------------------------------------===//
// Tensor Functionalities
//===----------------------------------------------------------------------===//


extern "C" void *view(int thread_id, Tensor *tensor, float first_dim, ...)
{
  //std::cout << "Executing: " << tensor.name << "." << "view" << "\n";
   
  std::vector<float> new_dims, new_dims_no_minus, current_dims;
  bool has_minus = false;
  current_dims = tensor->dims;

  
  va_list args;
  va_start(args, first_dim);

  if (first_dim!=-1)
    new_dims_no_minus.push_back(first_dim);
  else
    has_minus=true;
  
  
  new_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions??? (view)");
      return 0;
    }

    float dim = va_arg(args, float);
    if (dim==TERMINATE_VARARG)
      break;
    new_dims.push_back(dim);

    if (dim!=-1)
      new_dims_no_minus.push_back(dim);
    else
      has_minus=true;
  }
  va_end(args);


  


  int current_dims_prod = DimsProd(current_dims);
  int new_dims_prod = DimsProd(new_dims);


  if (has_minus)
  {
    float hidden_dim = (float)current_dims_prod / (float)DimsProd(new_dims_no_minus);

    if ((float)((int)hidden_dim) != hidden_dim)
    {
      LogErrorS("Automatic view dimension calculus resulted on a non-integer dimension.");
      PrintDims(current_dims);
      std::cout << "Current dims product: " << current_dims_prod  << ".\n";
      PrintDims(new_dims);
      std::cout << "New dims product: " << std::to_string(DimsProd(new_dims_no_minus))  << ".\n";
      return 0;
    }
    
    for (int i=0; i<new_dims.size(); i++)
      if (new_dims[i]==-1)
        new_dims[i] = hidden_dim;
    
  } else {
    if (current_dims_prod != new_dims_prod)
    {
      LogErrorS("Incompatible view dimensions.");
      PrintDims(current_dims);
      std::cout << "Current dims product: " << current_dims_prod  << ".\n";
      PrintDims(new_dims);
      std::cout << "New dims product: " << new_dims_prod  << ".\n";
      return 0;
    }
  }

  

  Tensor *new_tensor = createTensor(tensor->tensor_ptr, new_dims, DimsProd(new_dims), false, "");
  new_tensor->view_of = tensor->name;
  new_tensor->op=view_op;
  return new_tensor;
}






extern "C" void *NewVecToTensor(int thread_id, float first_dim, ...)
{
  std::vector<float> values;

  
  va_list args;
  va_start(args, first_dim);


  values.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("Tried to create a tensor from brackets with more than 10 positions. This is not yet supported");
      return nullptr;
    }

    float dim = va_arg(args, float);
    if (dim==TERMINATE_VARARG)
      break;
    values.push_back(dim);


  }
  va_end(args);


  float dims_prod = values.size();

  float *tensor_ptr, *tensor_cpu;
  tensor_cpu = values.data();

  tensor_ptr = get_from_pool(thread_id, dims_prod, "tensor from brackets");
  cudaMemcpy(tensor_ptr, tensor_cpu, dims_prod*sizeof(float), cudaMemcpyHostToDevice);
  

  Tensor *new_tensor = createTensor(tensor_ptr, {dims_prod}, dims_prod, true, "");
  new_tensor->op=create_tensor_from_brackets_op;
  return new_tensor;
}


//===----------------------------------------------------------------------===//
// Tensor -- Scalar   Operations
//===----------------------------------------------------------------------===//

std::vector<int> CalculateGridAndBlockSizes(int dims_prod, int pre_block_size=-1)
{

  int grid_size, block_size, shared_mem_size;

  if (pre_block_size==-1)
  {
    if (dims_prod<64)
      block_size = 32;
    else if (dims_prod<128)
      block_size = 64;
    else if (dims_prod<256)
      block_size = 128;
    else if (dims_prod<512)
      block_size = 256;
    else
      block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;
  } else
    block_size = pre_block_size;

  grid_size = ceil_div(dims_prod, block_size);
  shared_mem_size = 2 * block_size / 32 * sizeof(float);
  //shared_mem_size = std::min(2 * block_size * sizeof(float), deviceProp.sharedMemPerBlock);

  std::vector<int> ret = {grid_size, block_size, shared_mem_size};
  return ret;
}


std::vector<int> CalculateSimpleWarpGridAndBlockSizes(int B)
{
  // Usually warp kernels deal with the C dim already, so
  // you should inform B as the first dim only in these cases.

  int grid_size, block_size;

  block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;
  
  while (B < block_size/32 && block_size>32)
     block_size = block_size / 2;

  if (block_size<32)
    block_size = 32;

  grid_size = ceil_div(B, block_size/32);

  std::vector<int> ret = {grid_size, block_size};
  return ret;
}


__inline__ __device__ float cuda_clip(float val, float min_val, float max_val) {
    return fmaxf(min_val, fminf(val, max_val));
}








struct Grid {

  dim3 g;
  dim3 b;
  dim3 w;


  int smem;

  int wx_per_bx;
  int wy_per_by;

  void NewGrid(int gx, int gy, int bx, int by)
  {
    this->g.x = gx;
    this->g.y = gy;
    this->g.z = 1;

    this->b.x = bx;
    this->b.y = by;
    this->b.z = 1;

    smem = (bx+by)*32*sizeof(float);
  }

  void SetWarpSize(int wx, int wy)
  {
    wx_per_bx = b.x / (wx*16);
    wy_per_by = b.y / (wy*16);

    this->w.x = wx*32;
    this->w.y = wy;
    this->w.z = 1;
  }
};

Grid CalculateBlockingSize(int M, int N)
{

  int bx = 64;
  int by = 32;


  while(bx>M && bx>64)
    bx = bx/2;

  while(by>N && by>64)
    by = by/2;

  int gx = std::floor((M+bx-1)/(float)bx);
  int gy = std::floor((N+by-1)/(float)by);

  Grid grid;

  // std::cout << gx << ", " << gy << ", " << bx << ", " << by << "\n";
  grid.NewGrid(gx, gy, bx, by);


  int wx = fminf(fmaxf(M/16,1),4);
  int wy = fminf(fmaxf(N/16,1),4);

  wx = 4;
  wy = 2;

  grid.SetWarpSize(wx, wy);

  return grid;
}








__global__ void set_to_zero_kernel(float *y, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = 0;
}
__global__ void set_to_one_kernel(float *y, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = 1;
}

__global__ void set_to_minus_inf_kernel(float *y, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = -INFINITY;
}

__global__ void copy_tensor_kernel(float *y, const float *x, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = x[i];
}
__global__ void ema_tensor_kernel(float *y, const float *x, const float factor, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = factor*y[i] + (1-factor)*x[i];
}

__global__ void vec_mult(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = x[idx] * a;
  }
}
__global__ void vec_div(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = x[idx] / a;
  }
}
__global__ void vec_reverse_div(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = a / x[idx];
  }
}
__global__ void vec_add(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = x[idx] + a;
  }
}
__global__ void vec_sub(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = x[idx] - a;
  }
}
__global__ void vec_equal(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]==a) ? 1.0f : 0.0f;
  }
}
__global__ void vec_diff(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]!=a) ? 1.0f : 0.0f;
  }
}
__global__ void vec_higher(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]>a) ? 1.0f : 0.0f;
  }
}
__global__ void vec_higher_eq(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]>=a) ? 1.0f : 0.0f;
  }
}
__global__ void vec_minor(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]<a) ? 1.0f : 0.0f;
  }
}
__global__ void vec_minor_eq(const float a, float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = (x[idx]<=a) ? 1.0f : 0.0f;
  }
}
__global__ void vec_log(const float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = logf(x[idx]);
  }
}
__global__ void vec_log2(const float* x, float* y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = log2f(x[idx]);
  }
}

__global__ void tensor_div(float *w, float *x, float *y, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    y[idx] = x[idx] / w[idx];
  }
}

__global__ void tensor_clip(float* x, float *y, float _min, float _max, int dims_prod) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dims_prod) {
    if (x[idx]>_max)
      y[idx] = _max;
    else if (x[idx]<_min)
      y[idx] = _min;
    else
      y[idx] = x[idx];
  }
}



extern "C" void *CudaScalarMult(Tensor *tensor, float R, int thread_id) {
  //std::cout << "CudaScalarMult by " << R << "\n";
  
  int kDataLen = tensor->dims_prod;

  
  float* device_y = get_from_pool(thread_id, kDataLen, "scalar mult");
  

  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  
  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_mult<<<grid_size, block_size, 0, stream>>>(R, tensor->tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor->dims, kDataLen, false, "");
  new_tensor->AttrLNode(tensor, scalar_mult_op);
  new_tensor->scalar = R;
  return new_tensor;
}


extern "C" void *CudaScalarDiv(Tensor tensor, float R, int thread_id) {

  int kDataLen = tensor.dims_prod;


  
  float* device_y = get_from_pool(thread_id, kDataLen, "scalar div");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_div<<<grid_size, block_size, 0, stream>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  
  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}

extern "C" void *CudaReverseScalarDiv(Tensor tensor, float R, int thread_id) {

  int kDataLen = tensor.dims_prod;


  
  float* device_y = get_from_pool(thread_id, kDataLen, "reverse scalar div");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  
  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_reverse_div<<<grid_size, block_size, 0, stream>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}

extern "C" void *CudaScalarAdd(Tensor *tensor, float R, int thread_id) {
  
  int dims_prod = tensor->dims_prod;


  float* device_y = get_from_pool(thread_id, dims_prod, "scalar add");
  
  
  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  
  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_add<<<grid_size, block_size, 0, stream>>>(R, tensor->tensor_ptr, device_y, dims_prod);
  
  Tensor *new_tensor = createTensor(device_y, tensor->dims, dims_prod, false, "");
  new_tensor->AttrLNode(tensor, scalar_add_op);
  return new_tensor;
}

extern "C" void *CudaScalarSub(Tensor *tensor, float R, int thread_id) {

  int kDataLen = tensor->dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  
  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_sub<<<grid_size, block_size, 0, stream>>>(R, tensor->tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor->dims, kDataLen, false, "");
  new_tensor->AttrLNode(tensor, scalar_sub_op);
  return new_tensor;
}

extern "C" void *CudaScalarEqual(Tensor tensor, float R, int thread_id) {

  int kDataLen = tensor.dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  
  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_equal<<<grid_size, block_size, 0, stream>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}
extern "C" void *CudaScalarDiff(Tensor tensor, float R, int thread_id) {

  int kDataLen = tensor.dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_diff<<<grid_size, block_size, 0, stream>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}
extern "C" void *CudaScalarMinor(Tensor tensor, float R, int thread_id) {

  int kDataLen = tensor.dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_minor<<<grid_size, block_size, 0, stream>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}
extern "C" void *CudaScalarMinorEq(Tensor tensor, float R, int thread_id) {

  int kDataLen = tensor.dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_minor_eq<<<grid_size, block_size, 0, stream>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}
extern "C" void *CudaScalarHigher(Tensor tensor, float R, int thread_id) {

  int kDataLen = tensor.dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_higher<<<grid_size, block_size, 0, stream>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}
extern "C" void *CudaScalarHigherEq(Tensor tensor, float R, int thread_id) {

  int kDataLen = tensor.dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_higher_eq<<<grid_size, block_size, 0, stream>>>(R, tensor.tensor_ptr, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, tensor.dims, kDataLen, false, "");
  return new_tensor;
}


//TODONOW
extern "C" void *logE(int thread_id, Tensor tensor) {
  //std::cout << "logE of: " << tensor.name << "\n";

  float * device_x = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  int kDataLen = tensor.dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_log<<<grid_size, block_size, 0, stream>>>(device_x, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, dims, kDataLen, false, "");
  return new_tensor;
}

extern "C" void *logE2(int thread_id, Tensor tensor) {
  std::cout << "logE2 of: " << tensor.name << "\n";

  float * device_x = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  int kDataLen = tensor.dims_prod;


  float* device_y = get_from_pool(thread_id, kDataLen, "scalar sub");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(kDataLen);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor.Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  vec_log2<<<grid_size, block_size, 0, stream>>>(device_x, device_y, kDataLen);

  Tensor *new_tensor = createTensor(device_y, dims, kDataLen, false, "");
  return new_tensor;
}


extern "C" float CalculateIdxOffset(char *tensor_name, float first_idx, ...) {
  
  //std::cout << "CalculateIdxOffset of " << tensor_name << "\n";

  Tensor *tensor = NamedTensorsT[tensor_name];

  std::vector<float> idxs, new_dims_no_minus, dims;
  int current_dims_prod;
  bool has_minus = false;
  dims = tensor->dims;

  int idx_at = 0;

  
  va_list args;
  va_start(args, first_idx);

  if (first_idx!=-1)
    new_dims_no_minus.push_back(first_idx);
  else
    has_minus=true;
  
    
  idxs.push_back(first_idx);

  dims = RemoveFirstDim(dims);
  
  current_dims_prod = DimsProd(dims);

  idx_at += (int)(current_dims_prod*first_idx);



  //std::cout << "Get idx of " << tensor_name << "\nCalculateIdxOffset pushing dim: " << first_idx << "\n";

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions??? (calc idx)");
      return 0;
    }

    float idx = va_arg(args, float);
    if (idx==-40370000000)
      break;

    idxs.push_back(idx);
    
    dims = RemoveFirstDim(dims);
    
    current_dims_prod = DimsProd(dims);

    idx_at += (int)(current_dims_prod*idx);

    //std::cout << "CalculateIdxOffset pushing dim: " << idx << "\n";
    

    if (idx!=-1)
      new_dims_no_minus.push_back(idx);
    else
      has_minus=true;
  }
  va_end(args);



  return idx_at;
}


extern "C" void AttrPinnedOnIdx(char *tensor_name, float val, float idx_at) {
  Tensor *tensor = NamedTensorsT[tensor_name];
  //std::cout << "AttrPinnedOnIdx for " << tensor->name << " at index " << idx_at << "\n";
  //std::cout << "Value: " << val <<"\n";

  std::vector<float> dims = tensor->dims;
  int dims_prod = DimsProd(dims);
  if (idx_at>(dims_prod-1))
  {
    std::string _error = "\n\t- Idexating at pos: \033[32m"+std::to_string((int)idx_at);
    _error = _error + "\033[0m on pinned_tensor \033[95m"+std::string(tensor_name);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";
  }

  float *base_address = tensor->cpu_tensor_ptr;
  
  
  //std::cout << "idx " << idx_at << ", val " << val << "\n";

  float *device_x = base_address + static_cast<int>(idx_at);

  *device_x = val;
  move_to_char_pool(strlen(tensor_name)+1, tensor_name, "free");
}







extern "C" float AttrPinnedFromTensorOnIdx(char *tensor_name, Tensor *Rtensor, int thread_id, float first_idx, ...)
{
  
  //std::cout << "\n\n\nIDX " << tensor_name << "\n\n\n\n";  

  std::vector<float> idxs;

  va_list args;
  va_start(args, first_idx);

  idxs.push_back(first_idx);

  for (int i=0; i<10; i++)
  {
    float dim = va_arg(args, float);
    if (dim==TERMINATE_VARARG)
      break;
    idxs.push_back(dim);
  }
  va_end(args);  

  PrintDims(idxs);

  float offset = 0;
  std::vector<float> dims, aux_dims, Rdims;
  
  
  Tensor *tensor = NamedTensorsT[tensor_name];
  Rdims = Rtensor->dims;
  float R_dims_prod = Rtensor->dims_prod;

  float *new_tensor;

  dims = tensor->dims;
  std::vector<float> new_dims;

  if(idxs.size()>dims.size())
  {
    LogErrorS("The index used contain more dimensions than the indexed tensor.");
    return 0;
  }

  if (dims.size()==1)
    new_dims = {1.0f};
  else
  {
    aux_dims = dims;
    for (int i = 0; i < idxs.size(); i++)
    {
      aux_dims = RemoveFirstDim(aux_dims);
      offset += idxs[i]*DimsProd(aux_dims);
      std::cout << "ATTR INDEX DIMS_PROD IS " << DimsProd(aux_dims) << "\n";
    }
    new_dims = aux_dims;
  }


  int dims_prod = DimsProd(dims);
  if (offset>(dims_prod-1))
  {
    std::string _error = "\n\t- Idexating at pos: \033[32m"+std::to_string((int)offset);
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tensor_name);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return 0;
  }
  //std::cout << "IDX AT: " << offset << "\n";


  float *base_address = tensor->cpu_tensor_ptr;
  float *device_x = base_address + static_cast<int>(offset);



  for (int i=0; i<R_dims_prod; i++)
    device_x[i] = Rtensor->cpu_tensor_ptr[i];

  /*
  new_tensor = get_from_pool(thread_id, R_dims_prod, "idx tensor");
  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(R_dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  copy_tensor_kernel<<<grid_size, block_size, 0, stream>>>(device_x, Rtensor->cpu_tensor_ptr, R_dims_prod);
  


  Tensor *indexed = createTensor(new_tensor, new_dims, R_dims_prod, true, "");
  indexed->from_grad_or_load = tensor->from_grad_or_load;
  */
  return 0;
}




extern "C" char * FirstArgOnDemand(char *first_arg, char *pre_dotc, char *_class, char *method, int nested_function, int isSelf, int isAttribute)
{

  std::string _first_arg = first_arg;
  std::string pre_dot = pre_dotc;

  delete[] pre_dotc;


  //std::cout << "\n\n\nIncoming first arg: " << first_arg << " from pre-dot: " << pre_dot << ";\n   class: " << _class << ", method: " << method << "\n   is nested: " << nested_function <<".\n";
  //std::cout << "   is self: " << isSelf << ", is attribute: " << isAttribute << "\n\n\n";

  //std::cout << "\n\n\n";
  //for(auto& pair : NamedObjects)
  //  std::cout << "NamedObjects: " << pair.first << ": " << pair.second<< "\n";

  //TODO:
  
  if (!isSelf && isAttribute)
  {
    std::string ret = NamedObjects[pre_dot];
    //std::cout << "\nReturning " << ret << "\n\n\n\n";
    return str_to_char(ret);
  }
  
  
  
  
  if (pre_dot!="self")
  {
    if (nested_function)
      _first_arg = _first_arg+pre_dot;
    else
      _first_arg = pre_dot; 
  }

  return str_to_char(_first_arg);
}


extern "C" void InstantiateObject(char *scope, char *obj_name)
{
  //std::cout << "\n\n\n\nInstantiateObject of: " << scope << obj_name << "\n\n\n";
  std::string _obj_name = obj_name;

  NamedObjects[scope+_obj_name] = _obj_name + RandomString(13);
  //std::cout << "Saving " << NamedObjects[scope+_obj_name]  << "\n\n";
}


extern "C" char *objHash(char *scope, char *obj_name)
{
  std::string _obj_name = obj_name;
  std::string ret = NamedObjects[scope+_obj_name];
  return str_to_char(ret);
}


extern "C" char *LoadObject(char *obj_name)
{
  //std::cout << "LOADING OBJECT FROM " << obj_name << "\n";
  std::string ret = NamedObjects[obj_name];
  delete[] obj_name;
  //std::cout << "Load object of: " << ret << "\n";
  return str_to_char(ret);
}





std::map<size_t, std::vector<char *>> CharPool;

void move_to_char_pool(size_t length, char *char_ptr, std::string from)
{
  delete[] char_ptr;
  return;
  if (length==0)
    return;
  //std::cout << "\nmove_to_char_pool from: " << from << "\n";
  

  pthread_mutex_lock(&char_pool_mutex);
  std::vector<char *> chars_in_pool = CharPool[length];
  if (!in_char_ptr_vec(char_ptr, chars_in_pool))
  {
    //if(!(chars_in_pool.size()<30&&length==1))
    if(chars_in_pool.size()<270)
      CharPool[length].push_back(char_ptr);
    else
    {
      std::cout << "FREEING CHAR WITH length: " << length << " from: " << from <<  "\n";
      delete[] char_ptr;
    }
  } 
  pthread_mutex_unlock(&char_pool_mutex);
}

char *get_from_char_pool(size_t length, std::string from)
{
  if (length==0)
    return nullptr;


  char *char_ptr;
  char_ptr = (char*)malloc(length);
  return char_ptr;

  
  pthread_mutex_lock(&char_pool_mutex);
  if(CharPool.count(length)>0)
  {
    std::vector<char *> chars_in_pool = CharPool[length];
    if (chars_in_pool.size()>0)
    {
      //std::cout << "GETTING FROM CHAR POOL: " << length << "\n";
      char_ptr = chars_in_pool.back();
      CharPool[length].pop_back();
      pthread_mutex_unlock(&char_pool_mutex);
      return char_ptr;
    }
  }
  pthread_mutex_unlock(&char_pool_mutex);

  //std::cout << "\nMalloc new CHAR from " << from << " of size: " << length << "\n";

  char_ptr = (char*)malloc(length);
  return char_ptr;
}


extern "C" char *GetEmptyChar()
{
  char *empty_char = get_from_char_pool(1,"get empty char");
  empty_char[0] = '\0';
  return empty_char;
}

extern "C" void FreeCharFromFunc(char *_char, char *func) {
  std::cout << "FREEING " << _char << " at function: " << func << "\n";
  delete[] _char;
  std::cout << "freed" << "\n";
}


extern "C" void FreeChar(char *_char) {
  //std::cout << "FREEING " << _char << "\n";

  move_to_char_pool(strlen(_char)+1, _char, "free");
  //delete[] _char;
}






extern "C" char *CopyString(char *in_str)
{
  size_t length = strlen(in_str) + 1;
  char *copied = get_from_char_pool(length, "copy");
  memcpy(copied, in_str, length);

  //std::cout << "copy " << in_str << "\n";

  return copied;
}

extern "C" char * ConcatStr(char *lc, char *rc)
{
  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  char *result_cstr = get_from_char_pool(length_lc+length_rc, "concat");
  //char* result_cstr = new char[length_lc+length_rc]; 
  
  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);

  //std::cout << "ConcatStr " << result_cstr << "\n";

  return result_cstr;
}

extern "C" char * ConcatStrFreeLeft(char *lc, char *rc)
{
  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  //char* result_cstr = new char[length_lc+length_rc];
  char *result_cstr = get_from_char_pool(length_lc+length_rc, "concat free left");
  
  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);


  move_to_char_pool(length_lc+1, lc, "concat free left");
  //delete[] lc;

  //std::cout << "ConcatStrFreeLeft " << result_cstr << "\n";
  
  return result_cstr;
}

extern "C" char * ConcatStrFreeRight(char *lc, char *rc)
{
  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  //char* result_cstr = new char[length_lc+length_rc];
  char *result_cstr = get_from_char_pool(length_lc+length_rc, "concat free right");
  
  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);

  move_to_char_pool(length_rc, rc, "concat free right");
  //delete[] rc;

  //std::cout << "ConcatStrFreeRight " << result_cstr << "\n";
  
  return result_cstr;
}

extern "C" char * ConcatStrFree(char *lc, char *rc)
{
  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  char* result_cstr = new char[length_lc+length_rc]; 
  
  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);

  move_to_char_pool(length_lc+1, lc, "concat free");
  move_to_char_pool(length_rc, rc, "concat free");
  //delete[] lc, rc;
  
  //std::cout << "ConcatStrFree " << result_cstr << "\n";

  return result_cstr;
}


extern "C" char * ConcatFloatToStr(char *lc, float r)
{

  //TODO: Change and test the function below
  /*
    char buffer[32]; // 32 bytes should be enough to hold float as a string
    int len = snprintf(buffer, sizeof(buffer), "%.6f", r); // Format float as string with 6 decimal places

    // Calculate the total length for the result
    size_t lc_len = std::strlen(lc);
    size_t total_len = lc_len + len + 1; // +1 for null terminator

    // Allocate the result buffer
    char* result_cstr = new char[total_len];

    // Copy the input string and the formatted float to the result buffer
    std::memcpy(result_cstr, lc, lc_len);
    std::memcpy(result_cstr + lc_len, buffer, len + 1);
  */

  std::string l = lc;
  int _r = r;

  std::string result_str = l + std::to_string(_r);
  char* result_cstr = new char[result_str.length() + 1];
  std::strcpy(result_cstr, result_str.c_str());

  
  return result_cstr;
}

extern "C" char * ConcatNumToStrFree(char *lc, float r)
{
  //std::cout << "\nCONCAT NUM TO STR " << lc << " & " << std::to_string(r) << "\n";

  //TODO: Change and test the function below
  /*
    char buffer[32]; // 32 bytes should be enough to hold float as a string
    int len = snprintf(buffer, sizeof(buffer), "%.6f", r); // Format float as string with 6 decimal places

    // Calculate the total length for the result
    size_t lc_len = std::strlen(lc);
    size_t total_len = lc_len + len + 1; // +1 for null terminator

    // Allocate the result buffer
    char* result_cstr = new char[total_len];

    // Copy the input string and the formatted float to the result buffer
    std::memcpy(result_cstr, lc, lc_len);
    std::memcpy(result_cstr + lc_len, buffer, len + 1);
  */


  std::string l = lc;
  int _r = r;

  std::string result_str = l + std::to_string(_r);
  char* result_cstr = new char[result_str.length() + 1]; // +1 for null terminator
  std::strcpy(result_cstr, result_str.c_str());

  delete[] lc;
  
  return result_cstr;
}

extern "C" char * ConcatScopeStr(char *lc, char *rc)
{
  std::string lstr = lc;


  if (in_str(lstr, globalVars))
  {

    size_t length_lc = strlen(lc) + 1;
    //char* result_cstr = new char[length_lc+length_rc];
    char *result_cstr = get_from_char_pool(length_lc, "concat free left");
    memcpy(result_cstr, lc, length_lc);
    move_to_char_pool(length_lc+1, lc, "concat free left");
    return result_cstr;
  }
  

  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  //char* result_cstr = new char[length_lc+length_rc];
  char *result_cstr = get_from_char_pool(length_lc+length_rc, "concat free left");
  
  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);


  move_to_char_pool(length_lc+1, lc, "concat free left");
  //delete[] lc;

  //std::cout << "ConcatScopeStr " << result_cstr << "\n";
  
  return result_cstr;
}

extern "C" char * ConcatScopeAtCallExpr(char *lc, char *rc)
{
  //std::cout << "ConcatScopeAtCallExpr of " << lc << " and " << rc << "\n";
  std::string rstr = rc;

  //for (auto &a : globalVars)
  //  std::cout << "" << a << "\n";

  
  if (in_str(rstr, globalVars))
  {
    //std::cout << "it's a global" << "\n";
    size_t length_rc = strlen(rc) + 1; // +1 for null terminator
    char *result_cstr = get_from_char_pool(length_rc, "ConcatScopeAtCallExpr");
    memcpy(result_cstr, rc, length_rc);
    return result_cstr;
  }
  

  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  //char* result_cstr = new char[length_lc+length_rc];
  char *result_cstr = get_from_char_pool(length_lc+length_rc, "ConcatScopeAtCallExpr");
  
  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);

  
  return result_cstr;
}

extern "C" void AddFloatToScopeCleanList(char *scope, char *name)
{
  std::string _name, _scope;
  _name = name;
  _scope = scope;

  
  //std::cout << "will erase " << name << " from scope " << scope << "\n";
  pthread_mutex_lock(&clean_scope_mutex);
  ScopeVarsToClean[_scope].push_back(std::make_pair(_name, "float"));
  pthread_mutex_unlock(&clean_scope_mutex);
  
}

extern "C" void AddToScopeCleanList(char *scope, char *name)
{
  
  pthread_mutex_lock(&clean_scope_mutex);
  std::vector<std::pair<std::string, std::string>> scope_vars = ScopeVarsToClean[scope];
  
  for(auto &pair : scope_vars)
    if (pair.first==name)
    {
      delete[] name;
      return;
    }
    
  ScopeVarsToClean[scope].push_back(std::make_pair(name, "str"));
  pthread_mutex_unlock(&clean_scope_mutex);
  
  delete[] name;
}



void CleanThreadTensors(std::string, int);
void ThreadedCleanupToPool(Tensor *, std::string, int);

extern "C" void CleanScopeVars(char *scope, int thread_id)
{
  
  pthread_mutex_lock(&clean_scope_mutex);

  std::vector<std::pair<std::string, std::string>> scope_vars = ScopeVarsToClean[scope];

  for (auto _it = ScopeVarsToClean[scope].begin(); _it != ScopeVarsToClean[scope].end(); )
  {
    auto &pair = *_it;
    

    if (pair.second=="str")
    {
      NamedStrs.erase(pair.first);

      /*
      auto it = NamedStrs.find(pair.first);
      if (it != NamedStrs.end())
        NamedStrs.erase(it);
      */      
      
    }
    
    if (pair.second=="float")
    {
      NamedClassValues.erase(pair.first);
      /*
      auto it = NamedClassValues.find(pair.first);
      if (it != NamedClassValues.end())
        NamedClassValues.erase(it);
      */
    }
    
    _it = ScopeVarsToClean[scope].erase(_it);
  }

  if(thread_id!=0)
  {
    
    while(ThreadedScopeTensorsToClean[thread_id][scope].size()>0)
    {
      std::string tensor_name = ThreadedScopeTensorsToClean[thread_id][scope].back();
      ThreadedScopeTensorsToClean[thread_id][scope].pop_back();

      ThreadedCleanupToPool(NamedTensorsT[tensor_name], scope, thread_id);      
    }
    CleanThreadTensors(scope, thread_id);
    ThreadedScopeTensorsToClean[thread_id].erase(scope);
    
  }

  
  //ScopeVarsToClean[scope].clear(); // clear does not actually clears it
  auto it = ScopeVarsToClean.find(scope);
  ScopeVarsToClean.erase(it);

  pthread_mutex_unlock(&clean_scope_mutex);
  
}


extern "C" char * RandomStrOnDemand()
{ 
  return RandomString(14);
}


extern "C" void StoreOnDemandNoFree(char *name, float value){
  
  pthread_mutex_lock(&clean_scope_mutex);
  NamedClassValues[name] = value;
  pthread_mutex_unlock(&clean_scope_mutex);
}


extern "C" void StoreArgOnDemand(char *scope, char *name, float value){
  //std::cout << "StoreArgOnDemand: " << name  << " " << value << "\n";
  
  pthread_mutex_lock(&clean_scope_mutex);
  
  NamedClassValues[name] = value;

  std::string _name = name;
  
  ScopeVarsToClean[scope].push_back(std::make_pair(_name, "float"));
  pthread_mutex_unlock(&clean_scope_mutex);

  move_to_char_pool(strlen(name)+1, name, "free");
  //delete[] name;


}

extern "C" float StoreStrOnDemand(char *name, char *value){
  
  //NamedStrs[name] = CopyString(value); //TODO: Break?
  
  pthread_mutex_lock(&clean_scope_mutex);
  NamedStrs[name] = value;
  //std::cout << "Store " << value << " at " << name << "\n";
  pthread_mutex_unlock(&clean_scope_mutex);
  move_to_char_pool(strlen(name)+1, name, "free");
  //delete[] name;

  return 0;
}
extern "C" void *LoadStrOnDemand(char *name){
  
  //char *ret = CopyString(NamedStrs[name]);
  
  pthread_mutex_lock(&clean_scope_mutex);
  char *ret = NamedStrs[name];
  pthread_mutex_unlock(&clean_scope_mutex);
  move_to_char_pool(strlen(name)+1, name, "free");
  //delete[] name;

  return ret;
}


extern "C" float StoreStrVecOnDemand(char *name, std::vector<char *> value){
  //std::cout << "STORING " << self << "." << object_var_name << " on demand as StrVec type.\n";
  ClassStrVecs[name] = value;
  move_to_char_pool(strlen(name)+1, name, "free");
  //delete[] name;
  return 0;
}

extern "C" float StoreFloatVecOnDemand(char *name, std::vector<float> value){
  std::cout << "STORING " << name << " on demand as float vec type" << ".\n";

  ClassFloatVecs[name] = value;
  move_to_char_pool(strlen(name)+1, name, "free");
  //delete[] name;
  return 0;
}

extern "C" float StoreFloatVecOnDemandOnIdx(char *name, float idx, float value){
  //std::cout << "STORING " << self << "." << object_var_name << " on demand as float vec type" << ".\n";

  ClassFloatVecs[name][(int)idx] = value;
  move_to_char_pool(strlen(name)+1, name, "free");
  //delete[] name;
  return 0;
}




extern "C" void StoreOnDemand(char *name, float value){
  
  pthread_mutex_lock(&clean_scope_mutex);
  NamedClassValues[name] = value;
  pthread_mutex_unlock(&clean_scope_mutex);
  move_to_char_pool(strlen(name)+1, name, "StoreOnDemand");
  //delete[] name;
}
extern "C" float LoadOnDemand(char *object_var_name) {
  
  pthread_mutex_lock(&clean_scope_mutex);
  float ret = NamedClassValues[object_var_name];
  pthread_mutex_unlock(&clean_scope_mutex);
  
  move_to_char_pool(strlen(object_var_name)+1, object_var_name, "free");
  //delete[] object_var_name;
  return ret;
}

extern "C" float LoadOnDemandNoFree(char *object_var_name) {
  
  pthread_mutex_lock(&clean_scope_mutex);
  float ret = NamedClassValues[object_var_name];
  pthread_mutex_unlock(&clean_scope_mutex);
  
  return ret;
}




extern "C" void LockMutex(char *mutex_name)
{
  pthread_mutex_t *_mutex = lockVars[mutex_name];
  pthread_mutex_lock(_mutex);
}

extern "C" void UnlockMutex(char *mutex_name)
{
  pthread_mutex_t *_mutex = lockVars[mutex_name];
  pthread_mutex_unlock(_mutex);
}

extern "C" void *LoadStrVecOnDemand(char *object_var_name) {
  //std::cout << "Load StrVec On Demand var to load: " << object_var_name << "\n";
  
  void *ret = &ClassStrVecs[object_var_name];
  delete[] object_var_name;
  return ret;
}


extern "C" void *LoadFloatVecOnDemand(char *object_var_name) {
  std::cout << "Load StrVec On Demand var to load: " << object_var_name << "\n";
  
  void *ret = &ClassFloatVecs[object_var_name];
  delete[] object_var_name;
  return ret;
}




bool seen_var_attr = false;
Value *VariableExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look this variable up in the function.



  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  //std::string functionName = TheFunction->getName().str();
  

  //std::cout << "Create value V" << "\n";
  Value * ret = ConstantFP::get(*TheContext, APFloat(0.0f));
  Value *V, *var_name;



  /*
  std::cout << "\nVARIABLE EXPR CODEGEN: " << Name << "\n";
  for (const auto &entry : NamedStrVecs)
    std::cout << "NamedStrVec: " << entry.first << "\n";
  for (const auto &entry : NamedValues)
    std::cout << "NamedValues: " << entry.first << "\n";
  for (const auto &entry : NamedClassValues)
    std::cout << "NamedClassValues: " << entry.first << "\n";
  */

  std::string type = GetType();
  std::string pre_dot = GetPreDot();
  bool is_self = GetSelf();
  bool is_attr = GetIsAttribute();
  
  //std::string __print = "\n\nLOAD OF " + std::string(Name) + " ";
  //Builder->CreateCall(TheModule->getFunction("print"),
  //    {Builder->CreateGlobalString(__print), ConstantFP::get(*TheContext, APFloat(0.0f))});

  

  var_name = NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

  NameSolverAST *name_solver = static_cast<NameSolverAST *>(NameSolver.get());
  std::string Name = std::get<0>(name_solver->Names[name_solver->Names.size()-1]);
  

  if (is_self||is_attr)
  {
    if (type=="float")
    {
        V = Builder->CreateCall(TheModule->getFunction("LoadOnDemand"), {var_name});
        
        return V;
      
    }
    if (type=="str_vec")
    {
      V = Builder->CreateCall(TheModule->getFunction("LoadStrVecOnDemand"),
                                                      {var_name});
      //Builder->CreateCall(TheModule->getFunction("PrintStrVec"), {V});
      return V;
      
    }
    if (type=="float_vec"){
      V = Builder->CreateCall(TheModule->getFunction("LoadFloatVecOnDemand"),
                                                      {var_name});
      Builder->CreateCall(TheModule->getFunction("PrintFloatVec"), {V});
      return V;
    }
  }



  if (type=="float")
  {
    
    V = Builder->CreateCall(TheModule->getFunction("LoadOnDemand"), {var_name});
    

    //if (!seen_var_attr) //TODO: Solve this bug
    //  V = Builder->CreateCall(TheModule->getFunction("UnbugFloat"), {V}, "unbugfloat");

    return V;

  } else if (type=="object") {
    return var_name;

  } else if (type=="str") {

    
    //std::cout << "Type: " << Type << "\n\n";

    for (const auto &entry : NamedTensorsT)
    {
      std::cout << "Returning None because a tensor with name " << Name << " was found on strings map " << "\n";
      if (ends_with(entry.first, Name))
        return ret;
    }
    

    V = Builder->CreateCall(TheModule->getFunction("LoadStrOnDemand"), {var_name});

    //if (!seen_var_attr)
    //  Builder->CreateCall(TheModule->getFunction("PrintStr"), {V});
    
    return V;
  } else if (type=="str_vec") {

    //std::cout << "\nVariable Str Vector " << Name << " Codegen. \nNamedStrVecs.count(Name): " << NamedStrVecs.count(Name) <<"\n\n";



    V = NamedStrVecs[Name];
    
    V = Builder->CreateLoad(int8PtrTy, V, Name.c_str());
    if (!seen_var_attr)
      Builder->CreateCall(TheModule->getFunction("PrintStrVec"), {V});

    return V;
  } else if (NamedPinnedTensors.count(Name)>0) {
    //std::cout << "\nVariable Tensor " << Name << " Codegen.\n";
  

    if (!seen_var_attr)
      Builder->CreateCall(TheModule->getFunction("PrintTensor"), {thread_id, var_name});
    
    
    //Builder->CreateCall(TheModule->getFunction("PrintTensor"), {thread_id, var_name});

    return Builder->CreateCall(TheModule->getFunction("LoadTensor"), {var_name});
  } else if (type=="tensor") {
    //std::cout << "\nVariable Tensor " << Name << " Codegen.\n";


    if (!seen_var_attr)
    {
      Builder->CreateCall(TheModule->getFunction("PrintTensor"), {thread_id, var_name});
      return ConstantFP::get(*TheContext, APFloat(0.0f));
    }
    
    return Builder->CreateCall(TheModule->getFunction("LoadTensor"), {var_name});
  }
  else
  {
    std::string _error = "Variable " + Name + " does not exist";
    LogErrorS(_error);
  }
}






extern "C" char * AuxFn(char * arg1)
{

  std::cout << "Aux fn: " << arg1 << "\n";
  

  return arg1;
}





Value *VecIdxExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look this variable up in the function.

  std::cout << "Now Loading Vec indexation for type: " << Type << "  \n";


  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  
  
  Value * ret = ConstantFP::get(*TheContext, APFloat(0.0f));
  Value *V, *idx;

  if (Type!="object_vec")
    idx = Idx[0]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);


  Value *var_name, *object_name, *object_var_name;
  var_name = NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  
  NameSolverAST *name_solver = static_cast<NameSolverAST *>(NameSolver.get());
  std::string Name = std::get<0>(name_solver->Names[0]);
  


  std::string pre_dot = GetPreDot();
  bool is_self = GetSelf();
  bool is_attr = GetIsAttribute();
  std::cout << "is self: " << is_self << ", is_attr: " << is_attr << "\n";

  if (is_self||is_attr)
  {
    
    if (Type=="str_vec"){
      
      V = Builder->CreateCall(TheModule->getFunction("IndexClassStrVec"), {var_name, idx});
      
      return V;
    }

    if (Type=="float_vec"){
      V = Builder->CreateCall(TheModule->getFunction("IndexClassFloatVec"), {var_name, idx});
      return V;
    }

    if (Type=="object_vec")
      return var_name;
  }


  if (Type=="str_vec")
  {
    V = NamedStrVecs[Name];
    V = Builder->CreateLoad(int8PtrTy, V, Name.c_str());


    V = Builder->CreateCall(TheModule->getFunction("IndexStrVec"), {V, idx});

    return V;
  }
  if (Type=="float_vec")
  {
    V = NamedFloatVecs[Name];
    V = Builder->CreateLoad(int8PtrTy, V, Name.c_str());


    V = Builder->CreateCall(TheModule->getFunction("IndexStrVec"), {V, idx});

    return V;
  }

  if (Type=="tensor")
  {
    std::cout << "vec idx of tensor, idx type: " << Idx[0]->GetType() << "\n";

    if (Idx[0]->GetType()!="tensor")
    {
      /*
      std::vector<Value *> idx_calc_args;
      idx_calc_args.push_back(var_name);
      for (int i=0; i<Idx.size(); i++)
        idx_calc_args.push_back(Idx[i]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad));
      Value *idx_at = Builder->CreateCall(TheModule->getFunction("CalculateIdxOffset"),
                          idx_calc_args);

      return Builder->CreateCall(TheModule->getFunction("IdxTensor"), {var_name, idx_at, scope_str, thread_id});
      */
      std::vector<Value *> idx_calc_args;
      idx_calc_args.push_back(var_name);
      idx_calc_args.push_back(scope_str);
      idx_calc_args.push_back(thread_id);
      for (int i=0; i<Idx.size(); i++)
        idx_calc_args.push_back(Idx[i]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad));

      return Builder->CreateCall(TheModule->getFunction("IdxTensor"), idx_calc_args);
    } else {
      VariableExprAST *idx = static_cast<VariableExprAST *>(Idx[0].get());
      Value *idx_tensor_name = idx->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
      
      return Builder->CreateCall(TheModule->getFunction("IdxTensorWithTensor"), {var_name, idx_tensor_name, thread_id});
      
    }
    
  }

  std::string _error = "Unknown vector: " + Name + ".";
  LogErrorS(_error);
  std::cout << "Type " << Type << "\n";

  return ret;
}


Value *ObjectVecIdxExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look this variable up in the function.
  std::cout << "ObjectVecIdxExprAST codegen" << "\n";
  
  VecIdxExprAST *vec = static_cast<VecIdxExprAST *>(Vec.get());
  std::cout << "vec name " << vec->GetName() << "\n";
  std::cout << "ObjectVecIdxExprAST is vec: " << GetIsVec() << "\n";

  Value *idx = vec->Idx[0]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);


  Value *var_name, *object_name, *object_var_name, *post_dot_str;
  var_name = Builder->CreateGlobalString(vec->GetName());
  post_dot_str = Builder->CreateGlobalString(_post_dot);
  
  std::string pre_dot = GetPreDot();
  bool is_self = GetSelf();
  bool is_attr = GetIsAttribute();
  
  
  if (is_self||is_attr)
  {
    // Gets from pre_dot if it is a class attribute
    if (is_attr) {
      object_name = Builder->CreateGlobalString(pre_dot);
      var_name = Builder->CreateGlobalString(Name);

      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {object_name, var_name});
    }
    if (is_self)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {Builder->CreateLoad(int8PtrTy, first_arg), var_name});
  }

  if (Type=="tensor")
    return Builder->CreateCall(TheModule->getFunction("object_vec_idxTensor"),
                                                      {var_name, idx, post_dot_str});
  if (Type=="object")
    return Builder->CreateCall(TheModule->getFunction("object_vec_idxObject"),
                                                      {var_name, idx, post_dot_str});


  return ConstantFP::get(*TheContext, APFloat(0.0f));
}



__global__ void repeat_interleave_kernel_last_dim(const float *tensor,
                           float *probs,
                           int B, int C) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = threadIdx.x;
    
    if (i < B * C) {
        int b = i / (C);
        int v = i % C;

        float *probs_b = probs + b * C;
        float ix = tensor[b];

        probs_b[v] = ix;
    }
}




extern "C" void *IdxTensor(char *tensor_name, char *scope, int thread_id, float first_idx, ...)
{
  
  //std::cout << "\n\n\nIDX " << tensor_name << "\n\n\n\n";  

  std::vector<float> idxs;

  va_list args;
  va_start(args, first_idx);

  idxs.push_back(first_idx);

  for (int i=0; i<10; i++)
  {
    float dim = va_arg(args, float);
    if (dim==TERMINATE_VARARG)
      break;
    idxs.push_back(dim);
  }
  va_end(args);  

  PrintDims(idxs);

  float offset = 0;
  
  
  Tensor *tensor = NamedTensorsT[tensor_name];


  float *new_tensor;

  std::vector<float> dims, aux_dims;
  dims = tensor->dims;
  std::vector<float> new_dims;

  if(idxs.size()>dims.size())
  {
    LogErrorS("The index used contain more dimensions than the indexed tensor.");
    return nullptr;
  }

  if (dims.size()==1)
    new_dims = {1.0f};
  else
  {
    aux_dims = dims;
    for (int i = 0; i < idxs.size(); i++)
    {
      aux_dims = RemoveFirstDim(aux_dims);
      offset += idxs[i]*DimsProd(aux_dims);
      std::cout << "INDEX DIMS_PROD IS " << DimsProd(aux_dims) << "\n";
    }
    new_dims = aux_dims;
  }

  int new_dims_prod = DimsProd(new_dims);

  int dims_prod = DimsProd(dims);
  if (offset>(dims_prod-1))
  {
    std::string _error = "\n\t- Idexating at pos: \033[32m"+std::to_string((int)offset);
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tensor_name);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return nullptr;
  }
  //std::cout << "IDX AT: " << offset << "\n";


  float *base_address = tensor->tensor_ptr;
  float *device_x = base_address + static_cast<int>(offset);



  new_tensor = get_from_pool(thread_id, new_dims_prod, "idx tensor");
  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(new_dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  copy_tensor_kernel<<<grid_size, block_size, 0, stream>>>(new_tensor, device_x, new_dims_prod);
  

  /*
  PrintTensorF(new_tensor, 1, 1);
  PrintDims(new_dims);
  std::cout << "dims prod:" << new_dims_prod  << "\n";
  */

  if(nn_mode==eval_mode)
    ForwardCleanupToPool(tensor, scope);

  Tensor *indexed = createTensor(new_tensor, new_dims, new_dims_prod, true, "");
  indexed->from_grad_or_load = tensor->from_grad_or_load;
  return indexed;
}


__global__ void idx_last_dim_kernel(float *tgt,
                           const float *tensor, const float *idx_tensor, 
                           int dims_prod, int last_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int C = last_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *tgt_b = tgt + b;
        float idx_b = idx_tensor[b];

        if (v==idx_b)
        {
          float ix = tensor[i];
          tgt[b] = ix;
        }
    }
}




extern "C" void *IdxTensorWithTensor(char *tensor_name, char *idx_tensor_name, int thread_id)
{
  //std::cout << "INDEXATE TENSOR " << tensor_name << " WITH TENSOR " << idx_tensor_name << "\n";

  //std::cout << "\n\n\nIDX " << tensor_name << "\n\n\n\n";  
  
  
  Tensor *tensor = NamedTensorsT[tensor_name];
  Tensor *idx_tensor = NamedTensorsT[idx_tensor_name];


  float *tensor_ptr, *idx_tensor_ptr, *new_tensor;
  float new_dims_prod;
  std::vector<float> dims, idx_dims, new_dims;

  tensor_ptr = tensor->tensor_ptr;
  idx_tensor_ptr = idx_tensor->tensor_ptr;

  dims = tensor->dims;
  idx_dims = idx_tensor->dims;


  //TODO: gather with smaller dimensions
  /*
  if (dims.size()==1)
    new_dims = {1.0f};
  else
    for (int i = 0; i < dims.size()-1; i++)
      new_dims.push_back(dims[i+1]);
  */
  

  std::cout << "dim size diff: " << dims.size()-idx_dims.size()  << "\n";
  if((dims.size()-idx_dims.size())==1)
  {
    new_dims_prod = idx_tensor->dims_prod;
    new_dims = idx_tensor->dims;

    //std::cout << "INDEX OVER LAST DIM" << "\n";

    //cudaMalloc(&new_tensor, new_dims_prod*sizeof(float));
    //cudaMemset(new_tensor, 0, new_dims_prod*sizeof(float));

    new_tensor = get_from_pool(thread_id, new_dims_prod, "idx tensor with tensor");
    
    //int grid_size = tensor->dims_prod;
    //int block_size = 32;

    int grid_size, block_size;
    std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(tensor->dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];
    
    cudaStream_t stream = ThreadsStream[thread_id];
    idx_last_dim_kernel<<<grid_size, block_size, 0, stream>>>(new_tensor, tensor_ptr, idx_tensor_ptr, tensor->dims_prod, tensor->dims_prod/idx_tensor->dims_prod);
  }

  
  //cudaCheck(cudaMemcpy(new_tensor, device_x, new_dims_prod*sizeof(float), cudaMemcpyHostToHost));


  Tensor *indexed = createTensor(new_tensor, new_dims, new_dims_prod, false, "");
  indexed->AttrNodes(tensor, idx_tensor, idx_with_tensor_op);
  return indexed;
}



std::vector<std::string> seen_functions_for_pool_allocation;


extern "C" float CopyArgTensor(Tensor *tensor, char *new_tensor_name, char *previous_scope, char *scope, int thread_id)
{
  std::string tensor_name = tensor->name;
  //std::cout << "\n\n\nCOPY ARG TENSOR OF " << previous_scope << tensor_name << " into " << scope<<new_tensor_name  << " at thread: " << thread_id << "\n";

  
  
  std::string arg_tensor_name = scope;
  arg_tensor_name = arg_tensor_name + new_tensor_name;
  

  std::vector<float> dims = tensor->dims;
  int dims_prod = tensor->dims_prod;

  float *arg_tensor, *tensor_ptr;

  tensor_ptr = tensor->tensor_ptr;

  std::string _name = "arg tensor of ";
  _name = _name + tensor_name;
  arg_tensor = get_from_pool(thread_id, dims_prod, _name);
  
  //if (dims_prod!=0)//
  //  cudaMemcpy(arg_tensor, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToDevice);//
  
  if (dims_prod!=0)
  {
    int grid_size, block_size, shared_mem_size; 
    std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(tensor->dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];

    tensor->Sync();
    /*
    if (tensor->thread_id!=thread_id)
    {
      cudaStream_t prev_stream = ThreadsStream[tensor->thread_id];
      cudaStream_t stream = ThreadsStream[thread_id];

      cudaStreamSynchronize(prev_stream);

      copy_tensor_kernel<<<grid_size,block_size,0,stream>>>(arg_tensor, tensor_ptr, dims_prod);

      StreamAwaitStreamB(prev_stream, stream);
    } else {
      cudaStream_t stream = ThreadsStream[thread_id];
      copy_tensor_kernel<<<grid_size,block_size,0,stream>>>(arg_tensor, tensor_ptr, dims_prod);
    }
    */
    cudaStream_t stream = ThreadsStream[thread_id];
    copy_tensor_kernel<<<grid_size,block_size,0,stream>>>(arg_tensor, tensor_ptr, dims_prod);
  }
  

  Tensor *new_tensor = createTensor(arg_tensor, dims, dims_prod, true, arg_tensor_name, tensor->cuda_stream, tensor->loader);
  new_tensor->scopeless_name = tensor->scopeless_name;
  new_tensor->from_grad_or_load = tensor->from_grad_or_load;//
  NamedTensorsT[arg_tensor_name] = new_tensor;

  //if (thread_id!=0)
  //  ThreadedScopeTensorsToClean[thread_id][scope].push_back(arg_tensor_name);
  if (tensor->thread_id!=0)
  {
    //ThreadedScopeTensorsToClean[tensor->thread_id][previous_scope].erase(std::remove(ThreadedScopeTensorsToClean[tensor->thread_id][previous_scope].begin(), ThreadedScopeTensorsToClean[tensor->thread_id][previous_scope].end(), tensor_name), ThreadedScopeTensorsToClean[tensor->thread_id][previous_scope].end());
    threaded_Tensors_to_save[tensor->thread_id][previous_scope].push_back(tensor);
    threaded_tensors_to_save[tensor->thread_id][previous_scope].push_back(tensor->tensor_ptr);
  }

  return 0;
}


extern "C" float RemoveTensorScope(char *tensor_name, char *scope, char *tgt_tensorc, char *previous_scope, int thread_id)
{
  std::string tgt_tensor = tgt_tensorc;
  tgt_tensor = previous_scope + tgt_tensor;

  std::string scope_tensor_name = scope;
  scope_tensor_name = scope_tensor_name + tensor_name;


  //std::cout << "\n\n\nRETURNING " << scope_tensor_name << " into " << tgt_tensor << "\n\n\n\n";

  //std::cout << NamedTensorsT.count(tgt_tensor) <<  ", " << NamedTensorsT.count(scope_tensor_name) << "\n";
  
  Tensor *tensor, *scope_tensor;
  tensor = NamedTensorsT[tgt_tensor];


  if (tensor->thread_id != thread_id)
  {
    //std::cout << "\n\n\nRETURNING " << scope_tensor_name << " into " << tgt_tensor << "\n";
    std::cout << "Returning from thread id " << thread_id << " into " << tensor->thread_id << "\n\n\n\n";
    cudaStreamSynchronize(ThreadsStream[thread_id]);
    cudaStreamSynchronize(ThreadsStream[tensor->thread_id]);
  } else {
    //std::cout << "Returning from thread id " << thread_id << " into " << tensor->thread_id << "\n\n\n\n";
  }


  scope_tensor = NamedTensorsT[scope_tensor_name];
  std::vector<float> dims = scope_tensor->dims;
  int dims_prod = scope_tensor->dims_prod;


  std::string _name = "remove tensor scope of ";
  _name = _name + tensor_name;
  if(tensor->thread_id==0)
    move_to_pool(tensor->thread_id, tensor->dims_prod, tensor->tensor_ptr, _name);
  else
    ThreadedScopeTensorsToClean[tensor->thread_id][previous_scope].push_back(tensor->name);
  tensor->AttrTensor(scope_tensor->tensor_ptr, scope_tensor->dims, scope_tensor->dims_prod, scope_tensor->cuda_stream, scope_tensor->loader);
  tensor->from_grad_or_load = scope_tensor->from_grad_or_load;
  
  

  if(thread_id!=0)
  {
    //ThreadedScopeTensorsToClean[thread_id][scope].erase(std::remove(ThreadedScopeTensorsToClean[thread_id][scope].begin(), ThreadedScopeTensorsToClean[thread_id][scope].end(), scope_tensor_name), ThreadedScopeTensorsToClean[thread_id][scope].end());
    threaded_Tensors_to_save[thread_id][scope].push_back(scope_tensor);
    threaded_tensors_to_save[thread_id][scope].push_back(scope_tensor->tensor_ptr);
  }
  else if(nn_mode==eval_mode)//
    to_free_tensor_forward(scope_tensor, scope);//
  else
    to_free_tensor(scope_tensor);
  //delete scope_tensor;
  NamedTensorsT.erase(scope_tensor_name);

  scope_tensors[scope].clear();
  return 0;
}




extern "C" float RemoveTensorScopeAttrOnIndex(char *tensor_name, char *scope, char *tgt_tensorc, char *previous_scope, float idx_at, int thread_id)
{
  std::string scope_tensor_name = scope;
  scope_tensor_name = scope_tensor_name + tensor_name;

  std::string tgt_tensor = tgt_tensorc;
  tgt_tensor = previous_scope + tgt_tensor;


  std::cout << "\n\n\nRETURNING " << scope_tensor_name << " into " << tgt_tensor << " at idx\n\n\n\n";  



  Tensor *tensor, *scope_tensor;
  tensor = NamedTensorsT[tgt_tensor];

  scope_tensor = NamedTensorsT[scope_tensor_name];
  std::vector<float> dims = tensor->dims;
  int dims_prod = tensor->dims_prod;

  int scope_dims_prod = scope_tensor->dims_prod;

  
  if (idx_at>(dims_prod-1))
  {
    std::string _error = "\n\t- Idexating at pos: \033[32m"+std::to_string((int)idx_at);
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tgt_tensor);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return -1;
  }

  if ((idx_at+scope_dims_prod)>(dims_prod))
  {
    std::string _error = "\n\t- Attributing at pos: \033[32m"+std::to_string((int)idx_at)+"\033[0m with a tensor of size \033[32m"+std::to_string(scope_dims_prod)+"\033[0m";
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tgt_tensor);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return -1;
  }

  float *base_address = tensor->tensor_ptr;
  float *device_x = base_address + static_cast<int>(idx_at);

  cudaCheck(cudaMemcpy(device_x, scope_tensor->tensor_ptr, scope_dims_prod*sizeof(float), cudaMemcpyDeviceToDevice));
  
  return 0;
}



extern "C" float AttrTensor(char *tensor_name, Tensor *tensor, char *scope, int thread_id, int has_grad)
{
  //std::cout << "Attributing to tensor: " << tensor_name << " from " << tensor->name << "\n\n";

  //std::cout << "ATTRIBUTE TENSOR AT THREAD " << thread_id << "\n";

  Tensor *tgt_tensor = NamedTensorsT[tensor_name];
  
  
  if (tensor->view_of == tensor_name)
  {
    tgt_tensor->dims = tensor->dims;
    delete tensor;
  }
  else if (tensor->name==""||!tensor->leaf) // Free current and point to operation result
  {
    
    if(nn_mode==eval_mode||thread_id!=0)
    {
      if(tensor->from_grad_or_load) //if(DoesTreeContainWeight(tensor)>0)
        ForwardCleanupToPool(tensor, scope);
      ForwardCleanupToPool(tgt_tensor, scope);
      //scope_tensors[scope][tensor_name] = tensor->tensor_ptr;
      //else
      //  std::cout << "won't move to pool: " << tgt_tensor->name << ", " << tensor->op << ", " << tensor->name << "\n";
    }
    else {
      Tensor *attr_tensor;
      if (has_grad==0)
          tensor->op = detach_op;
      attr_tensor = createBackward(tgt_tensor->scopeless_name, tensor);
      todo_backward_tensors.push_back(attr_tensor);
    } 
    std::string scopeless_name = tgt_tensor->scopeless_name;
    tgt_tensor = createTensor(tensor->tensor_ptr, tensor->dims, tensor->dims_prod, true, tensor_name, tensor->cuda_stream, tensor->loader);
    tgt_tensor->from_grad_or_load = tensor->from_grad_or_load;
    tgt_tensor->scopeless_name = scopeless_name;
    
  } else { // Copy incoming tensor
  
  
    if(tensor->op==tensor_leaf||tensor->op==create_tensor_op||nn_mode==eval_mode||thread_id!=0)
    {
      if(tgt_tensor->dims != tensor->dims)
      {
        if(tgt_tensor->thread_id==0)
          move_to_pool(tgt_tensor->thread_id, tgt_tensor->dims_prod, tgt_tensor->tensor_ptr, "z=x");
        else
          ThreadedScopeTensorsToClean[tgt_tensor->thread_id][scope].push_back(tgt_tensor->name);

        tgt_tensor->tensor_ptr = get_from_pool(thread_id, tensor->dims_prod, "z=x");

        tgt_tensor->dims = tensor->dims;
        tgt_tensor->dims_prod = tensor->dims_prod;
      }
      //cudaCheck(cudaMemcpy(tgt_tensor->tensor_ptr, tensor->tensor_ptr, tgt_tensor->dims_prod*sizeof(float), cudaMemcpyDeviceToDevice));
      int grid_size, block_size, shared_mem_size; 
      std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(tensor->dims_prod);
      grid_size = grid_block_mem_sizes[0];
      block_size = grid_block_mem_sizes[1];

      tgt_tensor->Sync();
      tensor->Sync();

      /*
      if (tgt_tensor->thread_id!=thread_id)
      {
        cudaStream_t prev_stream = ThreadsStream[tgt_tensor->thread_id];
        cudaStream_t stream = ThreadsStream[thread_id];

        cudaStreamSynchronize(prev_stream);
        copy_tensor_kernel<<<grid_size,block_size,0,stream>>>(tgt_tensor->tensor_ptr, tensor->tensor_ptr, tensor->dims_prod);

        StreamAwaitStreamB(prev_stream, stream);

      } else {
        cudaStream_t stream = ThreadsStream[thread_id];
        copy_tensor_kernel<<<grid_size,block_size,0,stream>>>(tgt_tensor->tensor_ptr, tensor->tensor_ptr, tensor->dims_prod);
      }*/
      
      cudaStream_t stream = ThreadsStream[thread_id];
      copy_tensor_kernel<<<grid_size,block_size,0,stream>>>(tgt_tensor->tensor_ptr, tensor->tensor_ptr, tensor->dims_prod);

      if(nn_mode==training_mode&&thread_id==0)
      {
        Tensor *attr_tensor;
        
        //if (has_grad==0)
        //  tensor = wrapTensorWithDetached(tensor);

        if (has_grad==0)
          tensor->op = detach_op;  
        attr_tensor = createBackward(tgt_tensor->scopeless_name, tensor);
        todo_backward_tensors.push_back(attr_tensor);

        std::string scopeless_name = tgt_tensor->scopeless_name;
        tgt_tensor = createTensor(tgt_tensor->tensor_ptr, tensor->dims, tensor->dims_prod, true, tensor_name, tgt_tensor->cuda_stream, tgt_tensor->loader);
        
        tgt_tensor->from_grad_or_load = tensor->from_grad_or_load;
        tgt_tensor->scopeless_name = scopeless_name;
      } //else
        //scope_tensors[scope].push_back(tgt_tensor->tensor_ptr);
    } 
  }


  //if(nn_mode==eval_mode)
  //  CleanScopeTensors(scope);

  tgt_tensor->thread_id = thread_id;
  NamedTensorsT[tensor_name] = tgt_tensor;
  
  
  return 0;
}



extern "C" float print_scope(char *scope, char *previous_scope, int thread_id)
{

  std::cout << "\n- Scope is: " << scope << ";\n";
  std::cout << "- Previous scope was: " << previous_scope << ";\n";
  std::cout << "- Thread id: " << thread_id << ".\n\n";

  return 0;
}


// Copies a pinned_tensor's reserved memory into a tensor.
extern "C" float AttrTensorNoFree(char *tensor_name, Tensor *tensor, int thread_id)
{
  //std::cout << "\nAttrTensorNoFree -- Attributing to tensor: " << tensor_name << "\n\n";
  
  std::vector<float> new_dims = tensor->dims;
  float dims_prod = tensor->dims_prod;

  

  Tensor *tgt_tensor = NamedTensorsT[tensor_name];
  move_to_pool(tgt_tensor->thread_id, tgt_tensor->dims_prod, tgt_tensor->tensor_ptr, "pinned");
  

  //float *new_tensor;
  //cudaMalloc(&new_tensor, dims_prod*sizeof(float));
  float *new_tensor = get_from_pool(thread_id, dims_prod, "pinned");

  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor->Sync();
  copy_tensor_kernel<<<grid_size, block_size>>>(new_tensor, tensor->tensor_ptr, dims_prod);
  //cudaCheck(cudaMemcpy(new_tensor, tensor->tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToDevice));


  tgt_tensor->AttrTensor(new_tensor, new_dims, dims_prod);
  tgt_tensor->from_grad_or_load = tensor->from_grad_or_load;
  
  

  delete tensor;

  return 0;
}



extern "C" float AttrTensorOnIdx(char *tensor_name, Tensor *tensor, float idx_at, int thread_id)
{ 
  //std::cout << "AttrTensorOnIdx of" << tensor_name << " at idx " << idx_at << "\n";

  std::vector<float> dims, Rdims;
  Tensor *tgt_tensor = NamedTensorsT[tensor_name];
  dims = tgt_tensor->dims;
  int dims_prod = tgt_tensor->dims_prod;

  Rdims = tensor->dims;

  if (idx_at>(dims_prod-1))
  {
    std::string _error = "\n\t- Idexating at pos: \033[32m"+std::to_string((int)idx_at);
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tensor_name);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return -1;
  }

  int R_dims_prod = tensor->dims_prod;
  if ((idx_at+R_dims_prod)>(dims_prod))
  {
    std::string _error = "\n\t- Attributing at pos: \033[32m"+std::to_string((int)idx_at)+"\033[0m with a tensor of size \033[32m"+std::to_string(R_dims_prod)+"\033[0m";
    _error = _error + "\033[0m on tensor \033[95m"+std::string(tensor_name);
    _error = _error + "\033[0m;\n\t- Max idx allowed:  \033[32m"+std::to_string(dims_prod)+"\033[0m.";

    LogErrorS(_error);
    std::cout << "Dimensions:" << "\n";
    PrintDims(dims);
    std::cout << "\n";

    return -1;
  }

  float *base_address = tgt_tensor->tensor_ptr;
  float *device_x = base_address + static_cast<int>(idx_at);

  cudaStream_t stream = ThreadsStream[thread_id];
  //TODO*: turn into kernel
  cudaCheck(cudaMemcpyAsync(device_x, tensor->tensor_ptr, R_dims_prod*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    
  return 0;
}


__global__ void idx_attr_semi_last_dim_kernel(float *tgt,
                           const float *tensor, const float *idx_tensor, 
                           int dims_prod, int last_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int C = last_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *tgt_b = tgt + b;
        float idx_b = idx_tensor[b];

        if (v==idx_b)
        {
          float ix = tensor[b];
          tgt[i] = ix;
        }
    }
}


__global__ void idx_attr_simple_single_dim_kernel(float *tensor, const float *idx, const float *x, const int dims_prod)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid>=dims_prod)
    return; 

  tensor[(int)idx[tid]] = x[tid];
}

extern "C" float AttrTensorOnIdxTensor(char *tensor_name, char *idx_tensor_name, Tensor *R_tensor, int thread_id)
{ 
  //std::cout << "ATTR Idx tensor " << tensor_name << " at index tensor " << idx_tensor_name << " with tensor " << R_tensor->name << "\n";

  //std::cout << "\n\n\nIDX " << tensor_name << "\n\n\n\n";  
  
  
  Tensor *tensor = NamedTensorsT[tensor_name];
  Tensor *idx_tensor = NamedTensorsT[idx_tensor_name];


  float *tensor_ptr, *idx_tensor_ptr, *r_tensor_ptr;
  float dims_prod, new_dims_prod;
  std::vector<float> dims, idx_dims, new_dims;

  tensor_ptr = tensor->tensor_ptr;
  idx_tensor_ptr = idx_tensor->tensor_ptr;
  r_tensor_ptr = R_tensor->tensor_ptr;

  dims = tensor->dims;
  idx_dims = idx_tensor->dims;
  dims_prod = tensor->dims_prod;
  


  //TODO: gather with smaller dimensions
  /*
  if (dims.size()==1)
    new_dims = {1.0f};
  else
    for (int i = 0; i < dims.size()-1; i++)
      new_dims.push_back(dims[i+1]);
  */

  if (dims.size()<idx_dims.size())
  {
    LogErrorS("Index tensor must have less dimensions than the indexed tensor.");
    std::cout << "Tensor dims:" << "\n";
    PrintDims(dims);
    std::cout << "Idx tensor dims:" << "\n";
    PrintDims(idx_dims);
    return 0;
  }

  

  //std::cout << "dim size diff: " << dims.size()-idx_dims.size()  << "\n";

  cudaStream_t stream = ThreadsStream[thread_id];
  std::vector<int> grid_block_mem_sizes;
  int grid_size, block_size;

  
  //if((dims.size()-idx_dims.size())==0)
  if(dims.size()==1 && idx_dims.size()==1)
  {
    //std::cout << "INDEX ATTR OVER SIMPLE 1 DIM" << "\n";

    float idx_dims_prod = DimsProd(idx_dims);

    grid_block_mem_sizes = CalculateGridAndBlockSizes(idx_dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];

    //std::cout << "grid size: " << grid_size << " and block size: " << block_size << "\n";

    idx_attr_simple_single_dim_kernel<<<grid_size, block_size, 0, stream>>>(tensor_ptr, idx_tensor_ptr, r_tensor_ptr, idx_dims_prod);

  }
  if((dims.size()-idx_dims.size())==1)
  {
    //new_dims_prod = idx_tensor->dims_prod;
    //new_dims = idx_tensor->dims;

    std::cout << "INDEX ATTR OVER SEMI-LAST DIM" << "\n";

    grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];  

    idx_attr_semi_last_dim_kernel<<<grid_size, block_size, 0, stream>>>(tensor_ptr, r_tensor_ptr, idx_tensor_ptr, dims_prod, dims_prod/idx_tensor->dims_prod);
  }


  if(thread_id==0)
  {
    idx_tensor->op = detach_op;
    R_tensor->op = detach_op;
    todo_backward_tensors.push_back(idx_tensor);
    todo_backward_tensors.push_back(R_tensor);
  }

  return 0;
}





Value *BinaryTensorScalarExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Value *tensor_name = Builder->CreateGlobalString(LHS->GetName());



  std::string pre_dot = LHS->GetPreDot();
  bool is_self = LHS->GetSelf();
  bool is_attr = LHS->GetIsAttribute();

  if (is_attr) { // Gets from pre_dot if it is a class attribute
    Value * object_name = Builder->CreateGlobalString(pre_dot);

    tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {object_name, tensor_name});
  }
  if (is_self)
    tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {first_arg, tensor_name});
  if (!(is_self||is_attr))
    tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, tensor_name});
    



  // Special case '=' because we don't want to emit the LHS as an expression.
  if (Op == '=') {
    seen_var_attr=true;
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
    if (!LHSE)
      return LogErrorV("'=' destiny must be a var.");
    // Codegen the RHS.
    
    Value *Val = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
    if (!Val)
      return nullptr;

    
    
    std::cout << "1 0 attr\n";
    


    //LogErrorS("Attribution from float into tensor is not possible.");    
    
    
      
    
    seen_var_attr=false;
    return Val;
  }


  std::cout << "\n\n\nTensor scalar for LHS: " << LHS->GetName() << " RHS: " << RHS->GetName() << "\n\n\n";
  Value *LtensorPtr = LHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  Value *R = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  std::cout << "\n\n\nTensor scalar post codegen" << "\n\n\n";



  if (!LtensorPtr || !R)
    return nullptr;



  /*
  std::cout << "\nTensorScalar, LHS is self: " << LHS->GetSelf() << "\n";
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  std::cout << "Fname: " << functionName << "\n\n";
  */
  



  switch (Op)
  {
  case '*':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarMult"),
                               {LtensorPtr, R, thread_id}, "cudascalarmult");
  case '/':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarDiv"),
                               {LtensorPtr, R, thread_id}, "cudascalardiv");
  case 77:
    return Builder->CreateCall(TheModule->getFunction("CudaReverseScalarDiv"),
                               {LtensorPtr, R, thread_id}, "cudareversescalardiv");
  case '+':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarAdd"),
                               {LtensorPtr, R, thread_id}, "cudascalaradd");
  case '-':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarSub"),
                               {LtensorPtr, R, thread_id}, "cudascalarsub");
  case tok_equal:
    return Builder->CreateCall(TheModule->getFunction("CudaScalarEqual"),
                               {LtensorPtr, R, thread_id}, "cudascalarequal");
  case tok_diff:
    return Builder->CreateCall(TheModule->getFunction("CudaScalarDiff"),
                               {LtensorPtr, R, thread_id}, "cudascalardiff");
  case '<':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarMinor"),
                               {LtensorPtr, R, thread_id}, "cudascalarminor");
  case '>':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarHigher"),
                               {LtensorPtr, R, thread_id}, "cudascalarhigher");
  case tok_minor_eq:
    return Builder->CreateCall(TheModule->getFunction("CudaScalarMinorEq"),
                               {LtensorPtr, R, thread_id}, "cudascalarminoreq");
  case tok_higher_eq:
    return Builder->CreateCall(TheModule->getFunction("CudaScalarHigherEq"),
                               {LtensorPtr, R, thread_id}, "cudascalarhighereq");
  case ':':
    return LtensorPtr;
  case tok_space:
    return R;
  default:
    break;
  }
  

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {LtensorPtr, R};
  return Builder->CreateCall(F, Ops, "binop");
}



Value *BinaryPinnedScalarExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Value *tensor_name;



  

  if (Op == '=') {
    seen_var_attr=true;

    
    Value *Val = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
    if (!Val)
      return nullptr;
    
    std::cout << "2 0 attr\n";
    std::cout << "is vec: " << LHS->GetIsVec()  << "\n";


    


    VecIdxExprAST   *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
    tensor_name = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

    if (!LHSE)
      return LogErrorV("'=' destiny must be a variable.");

    

    std::vector<Value *> idx_calc_args;

    idx_calc_args.push_back(tensor_name);

    for (int i=0; i<LHSE->Idx.size(); i++)
    {
      idx_calc_args.push_back(LHSE->Idx[i]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad));
    }

    Value *idx_at = Builder->CreateCall(TheModule->getFunction("CalculateIdxOffset"),
                          idx_calc_args);

    Builder->CreateCall(TheModule->getFunction("AttrPinnedOnIdx"),
                          {tensor_name, Val, idx_at});


    /*
    if (LHS->GetIsVec())
    {
      VecIdxExprAST   *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
      if (!LHSE)
        return LogErrorV("'=' destiny must be a variable.");
      std::cout << "is vec: " << LHS->GetIsVec()  << "\n";

      Builder->CreateCall(TheModule->getFunction("AttrPinnedOnIdx"),
                          {tensor_name, LHSE->Idx->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad), Val});
    }
      
    else
    {
      VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
      if (!LHSE)
        return LogErrorV("'=' destiny must be a variable.");
    }
    */

    


    
    
    
      
    
    seen_var_attr=false;
    return Val;
  }


  
  Value *LtensorPtr = LHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  Value *R = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  



  
  
  if (!LtensorPtr || !R)
    return nullptr;



  /*
  std::cout << "\nTensorScalar, LHS is self: " << LHS->GetSelf() << "\n";
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  std::cout << "Fname: " << functionName << "\n\n";
  */
  



  switch (Op)
  {
  case '*':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarMult"),
                               {LtensorPtr, R, thread_id}, "cudascalarmult");
  case '/':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarDiv"),
                               {LtensorPtr, R, thread_id}, "cudascalardiv");
  case 77:
    return Builder->CreateCall(TheModule->getFunction("CudaReverseScalarDiv"),
                               {LtensorPtr, R, thread_id}, "cudareversescalardiv");
  case '+':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarAdd"),
                               {LtensorPtr, R, thread_id}, "cudascalaradd");
  case '-':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarSub"),
                               {LtensorPtr, R, thread_id}, "cudascalarsub");
  case ':':
    return LtensorPtr;
  case tok_space:
    return R;
  default:
    break;
  }
  

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {LtensorPtr, R};
  return Builder->CreateCall(F, Ops, "binop");
}







Value *BinaryPinnedAndTensorExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Value *tensor_name;



  

  if (Op == '=') {
    seen_var_attr=true;

    
    Value *RtensorPtr = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
    if (!RtensorPtr)
      return nullptr;
    
    std::cout << "2 0 attr\n";
    std::cout << "is vec: " << LHS->GetIsVec()  << "\n";


    


    VecIdxExprAST   *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
    tensor_name = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

    if (!LHSE)
      return LogErrorV("'=' destiny must be a variable.");

    

    std::vector<Value *> idx_calc_args;

    idx_calc_args.push_back(tensor_name);
    idx_calc_args.push_back(RtensorPtr);
    idx_calc_args.push_back(thread_id);

    for (int i=0; i<LHSE->Idx.size(); i++)
      idx_calc_args.push_back(LHSE->Idx[i]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad));


    Builder->CreateCall(TheModule->getFunction("AttrPinnedFromTensorOnIdx"),
                         idx_calc_args);


    /*
    if (LHS->GetIsVec())
    {
      VecIdxExprAST   *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
      if (!LHSE)
        return LogErrorV("'=' destiny must be a variable.");
      std::cout << "is vec: " << LHS->GetIsVec()  << "\n";

      Builder->CreateCall(TheModule->getFunction("AttrPinnedFromTensorOnIdx"),
                          {tensor_name, LHSE->Idx->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad), Val});
    }
      
    else
    {
      VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
      if (!LHSE)
        return LogErrorV("'=' destiny must be a variable.");
    }
    */

      
    
    seen_var_attr=false;
    return RtensorPtr;
  }


  
  Value *LtensorPtr = LHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  Value *R = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  



  
  
  if (!LtensorPtr || !R)
    return nullptr;




  switch (Op)
  {
  case '*':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarMult"),
                               {LtensorPtr, R, thread_id}, "cudascalarmult");
  case '/':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarDiv"),
                               {LtensorPtr, R, thread_id}, "cudascalardiv");
  case 77:
    return Builder->CreateCall(TheModule->getFunction("CudaReverseScalarDiv"),
                               {LtensorPtr, R, thread_id}, "cudareversescalardiv");
  case '+':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarAdd"),
                               {LtensorPtr, R, thread_id}, "cudascalaradd");
  case '-':
    return Builder->CreateCall(TheModule->getFunction("CudaScalarSub"),
                               {LtensorPtr, R, thread_id}, "cudascalarsub");
  case ':':
    return LtensorPtr;
  case tok_space:
    return R;
  default:
    break;
  }
  

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {LtensorPtr, R};
  return Builder->CreateCall(F, Ops, "binop");
}






extern "C" float min(float l, float r)
{
  if (l<r)
    return l;
  return r;
}
extern "C" float max(float l, float r)
{
  if (l>r)
    return l;
  return r;
}
extern "C" float logE2f(float v) {
  return log2f(v);
}
extern "C" float roundE(float v) {
  return round(v);
}
extern "C" float floorE(float v) {
  return floor(v);
}
extern "C" float logical_not(float v)
{
  if (v==0.0f)
    return 1;
  return 0;
}

extern "C" float dir_exists(char *path)
{
    if (std::filesystem::exists(path) && std::filesystem::is_directory(path))
    return 1;

  return 0;
}

extern "C" float path_exists(char *path)
{
  if (std::filesystem::exists(path))
    return 1;

  return 0;
}


extern "C" float randint(float b, float f)
{

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  float rand_float = dist(MAIN_PRNG);

  int rand_int = static_cast<int>(rand_float * (f - b + 1)) + b;

  return (float) rand_int;
}






__global__ void mult_backwarddx(const float *w,
                      float *dx, const float *dy,
                      const int tile_size, const int tile_offset,
                      const int B, const int C, const int OC) {

  int row = blockIdx.y * blockDim.y + threadIdx.y; // B
  int col = blockIdx.x * blockDim.x + threadIdx.x; // C
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  




  extern __shared__ char _smem[];
  auto smem = reinterpret_cast<float*>(_smem);



  int offset = tile_offset;

  float tmp = 0.0f;
  // consider row as B and col as C
  
  
#pragma unroll
  for (int i=0; i<ceilf(OC/(float)tile_size); ++i)
  {
    int _col = i*tile_size + tx;
    int _row = i*tile_size + ty;

    if( row<B  && _col<OC)
      smem[tx*tile_size +ty] = dy[row*OC + _col];      // [B, OC]
    else
      smem[tx*tile_size +ty] = 0;

    if(_row<OC &&  col<C)
      smem[offset+ty*tile_size +tx] = w[_row*C + col]; // [OC, C]
    else
      smem[offset+ty*tile_size +tx] = 0;
    
    __syncthreads();
    
#pragma unroll
    for(int j=0; j<tile_size; ++j)
      tmp += smem[j*tile_size +ty] * smem[offset+j*tile_size +tx];
    
    __syncthreads();
  }

  if(row<B && col<C)
    dx[row * C + col] = tmp;
}





__global__ void mult_backwarddw_acc(const float *x,
                      float *dw, const float *dy, const int tile_size, const int tile_offset,
                      int B, int C, int OC) {

  //int row_major = blockIdx.x * blockDim.x + threadIdx.y; // C
  //int col_major = blockIdx.x * blockDim.x + threadIdx.x; // OC

  int row = blockIdx.y * blockDim.y + threadIdx.y; // OC
  int col = blockIdx.x * blockDim.x + threadIdx.x; // C
  int tx = threadIdx.x;
  int ty = threadIdx.y;



  
  // backward type 1
  


  extern __shared__ char _smem[];
  auto smem = reinterpret_cast<float*>(_smem);



  // consider row as C and col as OC

  int offset = tile_offset;

  float tmp = 0.0f;
  // consider row as C and col as OC

  //int row = col_major%C; // Also, invert col_major with row_make BECAUSE I HAVE NO FKING IDEA WHY. 20 HOURS IMPLEMETNING THIS F MATRIX MULTIPLY TILING OPS. WHYYYYYYYYYYY????
  //int col = row_major%OC;

  // backward type 1

#pragma unroll
  for (int i=0; i<ceilf(B/(float)tile_size); ++i)
  {

    int _row  = i*tile_size + tx;
    int _row2 = i*tile_size + ty;

    if( _row<B  && row<OC)
      smem[tx*tile_size +ty] = dy[_row*OC + row];        // [B, OC]
    else
      smem[tx*tile_size +ty] = 0;

    if(_row2<B  && col<C)
      smem[offset+ty*tile_size +tx] = x[_row2*C + col];  // [B,  C]
    else
      smem[offset+ty*tile_size +tx] = 0;
    
    
    __syncthreads();
    
#pragma unroll
    for(int j=0; j<tile_size; ++j)
      tmp += smem[j*tile_size+ty] * smem[offset+j*tile_size+tx];
    
    __syncthreads();
  }

  if(col<C && row<OC)
    dw[row*C + col] += tmp;

  

  
  // backward type 2
  
  /*
  // consider row as C and col as OC
  if(col<OC && row<C)
  {
#pragma unroll
    for (int i=0; i<B; ++i)
      tmp += dy[i * OC + col] * x[i * C + row];
    dw[col * C + row] += tmp;
  }
  */
}


__global__ void mult_backwarddw(const float *x,
                      float *dw, const float *dy, const int tile_size, const int tile_offset,
                      int B, int C, int OC) {

  //int row_major = blockIdx.x * blockDim.x + threadIdx.y; // C
  //int col_major = blockIdx.x * blockDim.x + threadIdx.x; // OC

  int row = blockIdx.y * blockDim.y + threadIdx.y; // OC
  int col = blockIdx.x * blockDim.x + threadIdx.x; // C
  int tx = threadIdx.x;
  int ty = threadIdx.y;



  
  // backward type 1
  


  extern __shared__ char _smem[];
  auto smem = reinterpret_cast<float*>(_smem);



  // consider row as C and col as OC

  int offset = tile_offset;

  float tmp = 0.0f;
  // consider row as C and col as OC

  //int row = col_major%C; // Also, invert col_major with row_make BECAUSE I HAVE NO FKING IDEA WHY. 20 HOURS IMPLEMETNING THIS F MATRIX MULTIPLY TILING OPS. WHYYYYYYYYYYY????
  //int col = row_major%OC;

  // backward type 1

#pragma unroll
  for (int i=0; i<ceilf(B/(float)tile_size); ++i)
  {

    int _row  = i*tile_size + tx;
    int _row2 = i*tile_size + ty;

    if( _row<B  && row<OC)
      smem[tx*tile_size +ty] = dy[_row*OC + row];        // [B, OC]
    else
      smem[tx*tile_size +ty] = 0;

    if(_row2<B  && col<C)
      smem[offset+ty*tile_size +tx] = x[_row2*C + col];  // [B,  C]
    else
      smem[offset+ty*tile_size +tx] = 0;
    
    
    __syncthreads();
    
#pragma unroll
    for(int j=0; j<tile_size; ++j)
      tmp += smem[j*tile_size+ty] * smem[offset+j*tile_size+tx];
    
    __syncthreads();
  }

  if(col<C && row<OC)
    dw[row*C + col] = tmp;
}





void matmul_backward(float *inp,  float *weight,
                     int B, int C, int OC,
                     float *dinp, float *dw,
                     float *dout)
{
  
  // backward to input
  float one = 1.0f, zero = 0.0f;
  
  
  // backwad to dx
  cublasCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B, OC, &one,
                             weight, CUBLAS_LOWP, C, dout, CUBLAS_LOWP, OC, &zero,
                             dinp, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  
  
  // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
  cublasCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B, &one,
                             inp, CUBLAS_LOWP, C, dout, CUBLAS_LOWP, OC, &one,
                             dw, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  
  
  
  
  /*
  cudaStream_t dx_stream;
  cudaStreamCreate(&dx_stream);

  dim3 block_size(TILE_SIZE, TILE_SIZE);
  dim3 grid_size(std::ceil(C/(float)TILE_SIZE), std::ceil(B/(float)TILE_SIZE));
  int shared_mem_size = 2*TILE_SIZE_SQ*sizeof(float);

  cudaStreamSynchronize(main_stream);

  mult_backwarddx<<<grid_size, block_size, shared_mem_size>>>(weight, dinp, dout, TILE_SIZE, TILE_SIZE_SQ, B, C, OC);
  
  RegisterEvent(dx_stream);
  
  
  dim3 grid_size2(std::ceil(C/(float)TILE_SIZE), std::ceil(OC/(float)TILE_SIZE));
  mult_backwarddw_acc<<<grid_size2, block_size, shared_mem_size>>>(inp, dw, dout, TILE_SIZE, TILE_SIZE_SQ, B, C, OC);

  

  //PrintTensorF(dw, OC, C);

  StreamAwaitStreamB(main_stream, dx_stream);
  cudaStreamDestroy(dx_stream);
  */





  /*
  float alpha = 1.0f, beta = 1.0f;
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor = cutlass::layout::RowMajor;

  using CutlassGemm_dx = cutlass::gemm::device::Gemm<float,
                                                  RowMajor,
                                                  float,
                                                  RowMajor,
                                                  float,
                                                  RowMajor>;

  CutlassGemm_dx gemm_operator_dx;

  CutlassGemm_dx::Arguments args({B, C, OC},
                              {dout, OC},
                              {weight, C},
                              {dinp, C},
                              {dinp, C},
                              {alpha, beta});
                              
  gemm_operator_dx(main_stream);
  gemm_operator_dx(args);




  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor = cutlass::layout::RowMajor;

  using CutlassGemm_dw = cutlass::gemm::device::Gemm<float,
                                                  ColumnMajor,
                                                  float,
                                                  RowMajor,
                                                  float,
                                                  RowMajor>;

  CutlassGemm_dw gemm_operator_dw;

  CutlassGemm_dw::Arguments args_dw({OC, C, B},
                              {dout, OC},
                              {inp, C},
                              {dw, C},
                              {dw, C},
                              {alpha, beta});
                              
  gemm_operator_dw(main_stream);
  gemm_operator_dw(args_dw);
  */
}


__global__ void mult_kernel(const float *x, const float *w,
                      float *out, const int tile_size, const int tile_offset, const int B, const int C, const int OC) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x_block = blockIdx.x;
  int y_block = blockIdx.y;

  

  int row = y_block*tile_size + ty; // B
  int col = x_block*tile_size + tx; // OC



  int offset = tile_offset;

  float y = 0.0f;


  extern __shared__ float smem[];


  
#pragma unroll
  for (int i=0; i < ceilf(C/(float)tile_size); ++i)
  {

    int _col  = i * tile_size + tx;
    int _col2 = i * tile_size + ty;
    

    if(row<B && _col<C)
      smem[tx* tile_size +ty] = x[row*C + _col];
    else
      smem[tx* tile_size +ty] = 0;
    

    if (col<OC && _col2<C)
      smem[offset+ty* tile_size +tx] = w[col*C + _col2];
    else
      smem[offset+ty* tile_size +tx] = 0;
    
    __syncthreads();

#pragma unroll
    for(int j=0; j<tile_size; ++j)
      y += smem[j* tile_size +ty] * smem[offset+j* tile_size +tx];
    
    __syncthreads();
    
  }

  if(row<B && col<OC)
    out[row*OC+col] = y;
}









template<int WMMA_T, int X_WARPS, int Y_WARPS>
__global__ void wmma_backwarddx_kernel(float *dx, const float *w,
                      const float *dy, const int B, const int C, const int OC) {

  int laneId = ( threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
  int mw = laneId / WMMA_T;
  int ml = laneId % WMMA_T;

  int warp_y = threadIdx.y;
  int warp_x = (threadIdx.x / 32);


  const uint32_t warpX{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};  // C
  const uint32_t warpY{blockIdx.y * blockDim.y + threadIdx.y};               // B

  // warpX = (oc*X_WARPS + warp_x)




  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> y_frag;



  wmma::fill_fragment(y_frag, 0.0f);


  extern __shared__ float smem[];
  float *out_smem = smem;
  __half *hsmem = reinterpret_cast<__half*>(smem + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T));

  __half *x_smem     = hsmem;
  __half *w_smem     = hsmem + Y_WARPS*WMMA_T*(WMMA_T);
  
  
  

#pragma unroll
  for (int tile=0; tile<OC; tile+=WMMA_T)
  {

    /*
#pragma unroll
    for (int i=0; i<2; ++i)
    {
      // warp * mw_size * i_size + mw*i_size + i
      int row_aux1 = warp_x*((int)(warpSize/WMMA_T))*2 + mw*2+i;
      int row_aux2 = warp_y*((int)(warpSize/WMMA_T))*2 + mw*2+i;
      
      if (row_aux1<WMMA_T)
      {
        if ((warpY*WMMA_T+row_aux1)<B && (tile+ml)<OC)
          x_smem[(warp_y*WMMA_T+row_aux1)*WMMA_T + ml] = __float2half(*(dy + (warpY*WMMA_T+row_aux1)*OC + tile+ml));
        else
          x_smem[(warp_y*WMMA_T+row_aux1)*WMMA_T + ml] = 0;
      }

      if (row_aux2<WMMA_T)
      {
        if ((tile+ml)<OC && (warpX*WMMA_T+row_aux2)<C)
          w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = __float2half(*(w + (tile+ml)*C + warpX*WMMA_T+row_aux2));
        else
          w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = 0;
      }
    }
    */


    wmma::fill_fragment(x_frag, 0.0f);
    const auto func_x = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {

      if((warpY*WMMA_T+i)<B && (tile+j)<OC)
      {
        __half tmp = __float2half(*(dy + (warpY*WMMA_T+i)*OC + tile + j));
#pragma unroll
        for (unsigned f = 0; f < fragment_index_count; f++)
            x_frag.x[frag_index_list[f]] = tmp;
      } // else did not work, so fill_fragment is a workaround
    };
    
    __syncwarp();
    wmma_foreach_ij(
        x_frag,
        func_x
      );



    wmma::fill_fragment(w_frag, 0.0f);
    const auto func_w = [&](const unsigned* frag_index_list,
          const unsigned fragment_index_count,
          const unsigned i,
          const unsigned j) {
        
        if(((tile+i)<OC && warpX*WMMA_T+j)<C)
        { 
          __half tmp = __float2half(*(w + (tile+i)*C + warpX*WMMA_T+j));
  #pragma unroll
          for (unsigned f = 0; f < fragment_index_count; f++)
            w_frag.x[frag_index_list[f]] = tmp;
        }
      };

    __syncwarp();
    wmma_foreach_ij(
        w_frag,
        func_w
      );




    __syncwarp();
    if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<C)
    {
      //wmma::load_matrix_sync(x_frag, x_smem+(warp_y*WMMA_T)*WMMA_T, WMMA_T);
      //wmma::load_matrix_sync(w_frag, w_smem+(warp_x*WMMA_T)*WMMA_T, WMMA_T);
      wmma::mma_sync(y_frag, x_frag, w_frag, y_frag);
    }
    
  }


  if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<C && (warp_y*WMMA_T)<B && (warp_x*WMMA_T)<C)
  {

    float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;
    wmma::store_matrix_sync(_out, y_frag, X_WARPS*WMMA_T, wmma::mem_row_major);

    __syncthreads();
    
#pragma unroll
    for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)warpSize); ++tile)
    {
      int tile_idx = tile*warpSize + laneId;

      int row = tile_idx / WMMA_T;
      int col = tile_idx % WMMA_T;

      if((warpY*WMMA_T+row)<B  &&  (warpX*WMMA_T+col)<C && row<WMMA_T)
        dx[(warpY*WMMA_T+row)*C + warpX*WMMA_T+col] = _out[row*(X_WARPS*WMMA_T)+col];

    }
  }
}











template<int WMMA_T, int X_WARPS, int Y_WARPS>
__global__ void wmma_backwarddw_kernel(float *dw, const float *x,
                      const float *dy, const int B, const int C, const int OC) {

  int laneId = ( threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
  int mw = laneId / WMMA_T;
  int ml = laneId % WMMA_T;

  int warp_y = threadIdx.y;
  int warp_x = (threadIdx.x / 32);


  const uint32_t warpX{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};  // C
  const uint32_t warpY{blockIdx.y * blockDim.y + threadIdx.y};               // OC

  // warpX = (oc*X_WARPS + warp_x)




  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> y_frag;



  wmma::fill_fragment(y_frag, 0.0f);


  extern __shared__ float smem[];
  float *out_smem = smem;
  __half *hsmem = reinterpret_cast<__half*>(smem + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T));

  __half *x_smem     = hsmem;
  __half *w_smem     = hsmem + Y_WARPS*WMMA_T*(WMMA_T);
  
  
  

#pragma unroll
  for (int tile=0; tile<B; tile+=WMMA_T)
  {
    /*
#pragma unroll
    for (int i=0; i<2; ++i)
    {
      // warp * mw_size * i_size + mw*i_size + i
      int row_aux1 = warp_x*((int)(warpSize/WMMA_T))*2 + mw*2+i;
      int row_aux2 = warp_y*((int)(warpSize/WMMA_T))*2 + mw*2+i;
      
      if (row_aux1<WMMA_T)
      {
        if ((tile+ml)<B && (warpY*WMMA_T+row_aux1)<OC)
          x_smem[(warp_y*WMMA_T+row_aux1)*WMMA_T + ml] = __float2half(*(dy + (tile+ml)*OC + warpY*WMMA_T+row_aux1));
        else
          x_smem[(warp_y*WMMA_T+row_aux1)*WMMA_T + ml] = 0;
      }

      if (row_aux2<WMMA_T)
      {
        if ((tile+ml)<B && (warpX*WMMA_T+row_aux2)<C)
          w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = __float2half(*(x + (tile+ml)*C + warpX*WMMA_T+row_aux2));
        else
          w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = 0;
      }
    }
    */
    wmma::fill_fragment(x_frag, 0.0f);
    const auto func_x = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {

      if((tile+j)<B && (warpY*WMMA_T+i)<OC)
      {
        __half tmp = __float2half(*(dy + (tile+j)*OC + warpY*WMMA_T+i));
#pragma unroll
        for (unsigned f = 0; f < fragment_index_count; f++)
            x_frag.x[frag_index_list[f]] = tmp;
      } // else did not work, so fill_fragment is a workaround
    };
    
    __syncwarp();
    wmma_foreach_ij(
        x_frag,
        func_x
      );



    wmma::fill_fragment(w_frag, 0.0f);
    const auto func_w = [&](const unsigned* frag_index_list,
          const unsigned fragment_index_count,
          const unsigned i,
          const unsigned j) {
        
        if((tile+i)<B && (warpX*WMMA_T+j)<C)
        { 
          __half tmp = __float2half(*(x + (tile+i)*C + warpX*WMMA_T+j));
  #pragma unroll
          for (unsigned f = 0; f < fragment_index_count; f++)
            w_frag.x[frag_index_list[f]] = tmp;
        }
      };

    __syncwarp();
    wmma_foreach_ij(
        w_frag,
        func_w
      );



    __syncwarp();
    if ((warpY*WMMA_T)<OC && (warpX*WMMA_T)<C)
    {
      wmma::mma_sync(y_frag, x_frag, w_frag, y_frag);
    }
    
    
  }


  if ((warpY*WMMA_T)<OC && (warpX*WMMA_T)<C && (warp_y*WMMA_T)<OC && (warp_x*WMMA_T)<C)
  {
    float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;
    wmma::store_matrix_sync(_out, y_frag, X_WARPS*WMMA_T, wmma::mem_row_major);

    __syncthreads();
    
#pragma unroll
    for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)warpSize); ++tile)
    {
      int tile_idx = tile*warpSize + laneId;

      int row = tile_idx / WMMA_T;
      int col = tile_idx % WMMA_T;

      if((warpY*WMMA_T+row)<OC  &&  (warpX*WMMA_T+col)<C && row<WMMA_T)
        dw[(warpY*WMMA_T+row)*C + warpX*WMMA_T+col] = _out[row*(X_WARPS*WMMA_T)+col];

    }
  }
}












// matrix_a row_major
template <class Func>
__device__ inline void
wmma_foreach_ij(wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &frag,
           Func func) {

  const unsigned lane_id = threadIdx.x & 0x1f;
  const auto i_offset = lane_id / 4;
  const auto j_offset = (lane_id & 0b11) * 2;
  for (unsigned x = 0; x < frag.num_elements / 2; x++) {
    const unsigned i = i_offset + (x & 0b10) * 4;
    const unsigned j = j_offset + (x & 0b1) + (x & 0b100) * 2;
    const unsigned frag_index_list[2] = {x, x + 8};
    func(frag_index_list, 2, i, j);
  }
}



// matrix_a row_major other warp info
template <class Func>
__device__ inline void
wmma_foreach_ij(wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &frag,
           Func func, int other_warp, const int other_warp_max) {

  const unsigned lane_id = threadIdx.x & 0x1f;
  const auto i_offset = lane_id / 4;
  const auto j_offset = (lane_id & 0b11) * 2;

  for (unsigned tile = 0; tile < (unsigned)ceilf((frag.num_elements/2)/other_warp_max); tile++) {
    unsigned x = tile*other_warp_max + other_warp;

    const unsigned i = i_offset + (x & 0b10) * 4;
    const unsigned j = j_offset + (x & 0b1) + (x & 0b100) * 2;
    const unsigned frag_index_list[2] = {x, x + 8};
    func(frag_index_list, 2, i, j);
  }
}



// matrix_b col_major
template <class Func>
__device__ inline void
wmma_foreach_ij(wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                                  wmma::col_major> &frag,
           Func func) {
  const unsigned lane_id = threadIdx.x & 0x1f;
  const auto i_offset = (lane_id & 0b11) * 2;
  const auto j_offset = lane_id / 4;
  for (unsigned x = 0; x < frag.num_elements / 2; x++) {
    const unsigned i = i_offset + (x & 0b1) + (x & 0b10) * 4;
    const unsigned j = j_offset + (x & 0b100) * 2;
    const unsigned frag_index_list[2] = {x, x + 8};
    func(frag_index_list, 2, i, j);
  }
}



// accumulator
template <class T, class Func>
__device__ inline void
wmma_foreach_ij(wmma::fragment<wmma::accumulator, 16, 16, 16, T> &frag,
           Func func) {
  const unsigned lane_id = threadIdx.x & 0x1f;
  const unsigned i_offset = (lane_id >> 2);
  const unsigned j_offset = (lane_id & 0b11) * 2;
  for (unsigned x = 0; x < frag.num_elements; x++) {
    const unsigned j = j_offset + (x & 0b100) * 2 + (x & 0b1);
    const unsigned i = i_offset + (x & 0b10) * 4;
    const unsigned frag_index_list[1] = {x};
    func(frag_index_list, 1, i, j);
  }
}

// accumulator
template <class Func>
__device__ inline void
wmma_foreach_ij_acc(Func func) {
  const unsigned lane_id = threadIdx.x & 0x1f;
  const unsigned i_offset = (lane_id >> 2);
  const unsigned j_offset = (lane_id & 0b11) * 2;
  for (unsigned x = 0; x < 8; x++) {
    const unsigned j = j_offset + (x & 0b100) * 2 + (x & 0b1);
    const unsigned i = i_offset + (x & 0b10) * 4;
    const unsigned frag_index_list[1] = {x};
    func(frag_index_list, 1, i, j);
  }
}





template<int WMMA_T, int X_WARPS, int Y_WARPS>
__global__ void wmma_mult_kernel(const float *x, const float *w,
                      float *out, const int B, const int C, const int OC) {

  int laneId = ( threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
  int mw = laneId / WMMA_T;
  int ml = laneId % WMMA_T;

  int warp_y = threadIdx.y;
  int warp_x = (threadIdx.x / 32);


  const uint32_t warpX{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};  // OC
  const uint32_t warpY{blockIdx.y * blockDim.y + threadIdx.y};               // B

  // warpX = (oc*X_WARPS + warp_x)




  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> y_frag;

  using FRAG_T = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;



  wmma::fill_fragment(y_frag, 0.0f);


  extern __shared__ float smem[];
  float *out_smem = smem;
  __half *hsmem = reinterpret_cast<__half*>(smem + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T));

  __half *x_smem     = hsmem;
  __half *w_smem     = hsmem + Y_WARPS*WMMA_T*(WMMA_T);
  
  
  

#pragma unroll
  for (int tile=0; tile<C; tile+=WMMA_T)
  {

#pragma unroll
    for (int i=0; i<2; ++i)
    {
      // warp * mw_size * i_size + mw*i_size + i
      int row_aux1 = warp_x*((int)(warpSize/WMMA_T))*2 + mw*2+i;
      int row_aux2 = warp_y*((int)(warpSize/WMMA_T))*2 + mw*2+i;
      
      if (row_aux1<WMMA_T)
      {
        if ((warpY*WMMA_T+row_aux1)<B && (tile+ml)<C)
          x_smem[(warp_y*WMMA_T+row_aux1)*WMMA_T + ml] = __float2half(*(x + (warpY*WMMA_T+row_aux1)*C + tile+ml));
        else
          x_smem[(warp_y*WMMA_T+row_aux1)*WMMA_T + ml] = 0;
      }

      if (row_aux2<WMMA_T)
      {
        if ((warpX*WMMA_T+row_aux2)<OC && (tile+ml)<C)
          w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = __float2half(*(w + (warpX*WMMA_T+row_aux2)*C + tile+ml));
        else
          w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = 0;
      }
    }
    

    __syncthreads();


    if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC)
    {

      //wmma::fill_fragment(x_frag, 0.0f);

      
      

      /*
      asm volatile(".reg .b32 x<8>;"
                  "wmma.load.a.sync.aligned.m16n16k16.shared.row.f16"
                  " {x0, x1, x2, x3, x4, x5, x6, x7}, [%0], 16;"
                  :    // output registers
                  : "r"(x_smem+warp_y*WMMA_T*WMMA_T));
      */



      /*
      asm volatile(".reg .b32 x<8>;"
                   ".reg .b32 w<8>;"
                  "wmma.load.a.sync.aligned.m16n16k16.shared.row.f16"
                  " {x0, x1, x2, x3, x4, x5, x6, x7}, [%0], 16;"
                  "wmma.load.b.sync.aligned.m16n16k16.shared.col.f16"
                  " {w0, w1, w2, w3, w4, w5, w6, w7}, [%1], 16;"
                  :    // output registers
                  : "r"(x_smem+warp_y*WMMA_T*WMMA_T), "r"(w_smem+warp_x*WMMA_T*WMMA_T));
      */




      wmma::load_matrix_sync(x_frag, x_smem+warp_y*WMMA_T*WMMA_T, WMMA_T);



      /*
      asm volatile(".reg .b32 x<8>;"
                  "wmma.load.a.sync.aligned.m16n16k16.shared.row.f16"
                  " {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], 16;"
                  : "=r"(x_frag.x[0]), "=r"(x_frag.x[1]), "=r"(x_frag.x[2]), "=r"(x_frag.x[3]), "=r"(x_frag.x[4]), "=r"(x_frag.x[5]), "=r"(x_frag.x[6]), "=r"(x_frag.x[7])
                  : "r"(x_smem+warp_y*WMMA_T*WMMA_T));
      */
      


      wmma::load_matrix_sync(w_frag, w_smem+warp_x*WMMA_T*WMMA_T, WMMA_T);



      wmma::mma_sync(y_frag, x_frag, w_frag, y_frag);



      //y_frag.x[0] = x_frag.num_elements;
    }
    
    // __syncthreads();
  }


  if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC && (warp_y*WMMA_T)<B && (warp_x*WMMA_T)<OC)
  { 

    float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;
    wmma::store_matrix_sync(_out, y_frag, X_WARPS*WMMA_T, wmma::mem_row_major);

    // __syncthreads();
    
#pragma unroll
    for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)warpSize); ++tile)
    {
      int tile_idx = tile*warpSize + laneId;

      int row = tile_idx / WMMA_T;
      int col = tile_idx % WMMA_T;

      // if ((blockIdx.y+ warp_y+laneId)==0)
      //     printf("warpX: %d\t warpX offset: %d\t OC: %d\n", warpX, warpX*WMMA_T, OC);

      if((warpY*WMMA_T+row)<B  &&  (warpX*WMMA_T+col)<OC && row<WMMA_T)
        out[(warpY*WMMA_T+row)*OC + warpX*WMMA_T+col] = _out[row*(X_WARPS*WMMA_T)+col];

    }
  }
}

















__inline__ __device__ uint32_t cast_smem_ptr_to_uint(void* smem_ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

__inline__ __device__ int smem_xor_cp_async(int lane_id) {
    // *4 means we calculate the storage for 16B/128b words (4 floats), done for each thread
    return ((lane_id % 8) ^ (lane_id / 8) + (lane_id / 8)*8) * 4;
}


__inline__ __device__ int smem_dexor_from_cp_async(int strided, int contiguous) {
    int stride=8;
    
    int tc = contiguous / 8;
    int ts = strided / 4;

    int c = contiguous % 8;
    int s = strided % 4;

    int k_index = c / 2;

    int bank = ((c & 1) * 4) | (s ^ k_index); // e [0, 7]
    
    int offset = tc * 32 + bank + (ts * 4 + k_index) * stride;


    // *4 means we calculate the storage for 16B/128b words (4 floats), done for each thread
    return offset*4;
}



template<int WMMA_T, int X_WARPS, int Y_WARPS>
__global__ void wmma_cp_async(const float *x, const float *w,
                      float *out, const int B, const int C, const int OC) {

  int laneId = ( threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
  int mw = laneId / 4;
  int ml = laneId % 4;

  int warp_y = threadIdx.y;
  int warp_x = (threadIdx.x / 32);


  const uint32_t warpX{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};  // OC
  const uint32_t warpY{blockIdx.y * blockDim.y + threadIdx.y};               // B

  // warpX = (oc*X_WARPS + warp_x)




  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> y_frag;

  



  wmma::fill_fragment(y_frag, 0.0f);


  extern __shared__ float smem[];
  float *out_smem = smem;
  // __half *hsmem = reinterpret_cast<__half*>(smem + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T));

  // __half *x_smem     = hsmem;
  // __half *w_smem     = hsmem + Y_WARPS*WMMA_T*(WMMA_T);

  float *hsmem = smem;// + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T);

  float *x_smem     = hsmem;
  float *w_smem     = hsmem + Y_WARPS*WMMA_T*(WMMA_T*2);
  
  
  int k_count=0;
  int k_stride;

  int xor_addr = smem_xor_cp_async(laneId);

#pragma unroll
  for (int tile=0; tile<C; tile+=WMMA_T)
  {
    // warp * mw_size * i_size + mw*i_size + i
    
    k_stride = k_count % 2;
    k_count++;




    
    // each block deals with a 64x16 tile
    

    int row_aux1 = warp_x*4 + ml;
    int row_aux2 = warp_y*4 + ml;
    
    
    if (k_stride==0) { // loads 2 strides simultaneously


      // if ((warpX+warpY+laneId)==0)
      // {
      //   for (int i=0; i<32; ++i)
      //   {
      //     printf("%d\t", (int)smem_xor_cp_async(i)/4);
      //     if((i+1)%8==0)
      //       printf("\n");
      //   }
      //   printf("\n\n");
      // }

      if (row_aux1<WMMA_T)
      {
        float const *gmem_ptr = x + (warpY*WMMA_T+row_aux1)*C + tile+mw*4;
        
        // extra *2 to accomodate 32 instead of 16 C (i.e, the whole warpSize)
        //       *4 is necessary as it needs to accomodate 4 consecutive floats
        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&x_smem[(warp_y*WMMA_T+ warp_x*4)*WMMA_T*2 + xor_addr]);
        
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
                      :: "r"(smem_int_ptr),
                         "l"(gmem_ptr),
                         "n"(16),
                         "r"(((warpY*WMMA_T+row_aux1)<B) ? std::min((( C-(tile+mw*4)) /4)*4, 16) : 0)); // incorrect 0 padding yet
                         // "r"(((warpY*WMMA_T+row_aux1)<B) ? 16 : 0)); // incorrect 0 padding yet
        
      }

      if (row_aux2<WMMA_T)
      {
        // w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = *(w + (warpX*WMMA_T+row_aux2)*C + tile+ml);
        
        float const *gmem_ptr = w + (warpX*WMMA_T+row_aux2)*C + tile+mw*4;
        
        // extra 2 to accomodate 32 instead of 16 C
        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&w_smem[(warp_x*WMMA_T+ warp_y*4)*WMMA_T*2 + xor_addr]);
        
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
                      :: "r"(smem_int_ptr),
                         "l"(gmem_ptr),
                         "n"(16),
                         "r"(((warpX*WMMA_T+row_aux2)<OC) ? std::min((( C-(tile+mw*4)) /4)*4, 16) : 0));
                         // "r"(((warpX*WMMA_T+row_aux2)<OC) ? 16 : 0));
        
      }
    }
    // asm volatile("cp.async.commit_group;\n" ::);
    
    asm volatile("cp.async.wait_all;");
    
    __syncthreads();

    // if ((warpX+warpY+laneId)==0)
    // {
    //   for (int i=0; i<Y_WARPS*WMMA_T*32; ++i)
    //   {
    //     printf("%.2f, ", x_smem[i]*10000);
    //     if((i+1)%32==0)
    //       printf("\n");
    //   }
    //   printf("\n\n");
    // }

    if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC)
    {




      // if ((warpX+warpY+laneId)==0)
      // {
      //   for (int i=0; i<4*8; ++i)
      //   {
      //     int aux_i = i/8;
      //     int aux_j = i%8;
      //     printf("%d\t", (int)smem_dexor_from_cp_async(aux_i, aux_j)/4);
      //     if((i+1)%8==0)
      //       printf("\n");
      //   }
      //   printf("\n\n");
      // }


        
      
      const auto func_x = [&](const unsigned* frag_index_list,
          const unsigned fragment_index_count,
          const unsigned i,
          const unsigned j) {


          int wi = i/4;
          int xi = i%4;

          int xj = j/4;
          int wj = j%4;

          int offset = smem_dexor_from_cp_async(xi, xj*2 + k_stride)+wj;

        
          __half tmp = __float2half(*(x_smem + (warp_y*WMMA_T+ wi*4)*WMMA_T*2 + offset));


  #pragma unroll
          for (unsigned f = 0; f < fragment_index_count; f++)
              x_frag.x[frag_index_list[f]] = tmp;
      };
      
      __syncwarp();
      wmma_foreach_ij(
          x_frag,
          func_x
        );


      // if(warpY==0&&warpX==0&&laneId==0)
      //   printf("\n\n");


      
      const auto func_w = [&](const unsigned* frag_index_list,
            const unsigned fragment_index_count,
            const unsigned i,
            const unsigned j) {
          

            int wj = j/4;
            int xj = j%4;
          
            int xi = i/4;
            int wi = i%4;


            int offset = smem_dexor_from_cp_async(xj, xi*2+k_stride)+wi;

          
            __half tmp = __float2half(*(w_smem + (warp_x*WMMA_T+wj*4)*WMMA_T*2 + offset));
    #pragma unroll
            for (unsigned f = 0; f < fragment_index_count; f++)
              w_frag.x[frag_index_list[f]] = tmp;
        };

      __syncwarp();
      wmma_foreach_ij(
          w_frag,
          func_w
        );


      wmma::mma_sync(y_frag, x_frag, w_frag, y_frag);


    }
    
  }

  __syncthreads();

  if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC && (warp_y*WMMA_T)<B && (warp_x*WMMA_T)<OC)
  { 

    float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;
    
    wmma::store_matrix_sync(_out, y_frag, X_WARPS*WMMA_T, wmma::mem_row_major);
  //   const auto func_y = [&](const unsigned* frag_index_list,
  //         const unsigned fragment_index_count,
  //         const unsigned i,
  //         const unsigned j) {
                  
          
  // #pragma unroll
  //         for (unsigned f = 0; f < fragment_index_count; f++)
  //           _out[i*X_WARPS*WMMA_T + j] = y_frag.x[frag_index_list[f]];
  //     };

  //   __syncwarp();
  //   wmma_foreach_ij(
  //       y_frag,
  //       func_y
  //     );








    
#pragma unroll
    for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)(warpSize)); ++tile)
    {
      int tile_idx = tile*warpSize + laneId;

      int row =  tile_idx / WMMA_T;
      int col = (tile_idx % WMMA_T);


      if((warpY*WMMA_T+row)<B  &&  (warpX*WMMA_T+col)<OC && row<WMMA_T)
        out[(warpY*WMMA_T+row)*OC + warpX*WMMA_T+col] = _out[row*(X_WARPS*WMMA_T)+col];

    }
  }
}




















  
__inline__ __device__ void wmma16x16x16(wmma::fragment<wmma::accumulator, 16, 16, 16, float> &y_frag,
                                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &x_frag,
                                        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> &w_frag)
{
        const __half *X = reinterpret_cast<const __half *>(&x_frag.x);
        const __half *W = reinterpret_cast<const __half *>(&w_frag.x);
        
        asm volatile("wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32"
                     "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
                     "{%8,  %9,  %10, %11, %12, %13, %14, %15},"
                     "{%16, %17, %18, %19, %20, %21, %22, %23},"
                     "{%24, %25, %26, %27, %28, %29, %30, %31};"
                     : "=f"(y_frag.x[0]), "=f"(y_frag.x[1]), "=f"(y_frag.x[2]), "=f"(y_frag.x[3]), "=f"(y_frag.x[4]), "=f"(y_frag.x[5]), "=f"(y_frag.x[6]), "=f"(y_frag.x[7])
                     :  "r"(X[0]),  "r"(X[2]),  "r"(X[4]),  "r"(X[6]),  "r"(X[8]),  "r"(X[10]), "r"(X[12]), "r"(X[14]), \
                        "r"(W[0]),  "r"(W[2]),  "r"(W[4]),  "r"(W[6]),  "r"(W[8]),  "r"(W[10]), "r"(W[12]), "r"(W[14]), \
                       "f"(y_frag.x[0]), "f"(y_frag.x[1]), "f"(y_frag.x[2]), "f"(y_frag.x[3]), "f"(y_frag.x[4]), "f"(y_frag.x[5]), "f"(y_frag.x[6]), "f"(y_frag.x[7]));
        
        asm volatile("wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32"
                     "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
                     "{%8,  %9,  %10, %11, %12, %13, %14, %15},"
                     "{%16, %17, %18, %19, %20, %21, %22, %23},"
                     "{%24, %25, %26, %27, %28, %29, %30, %31};"
                     : "=f"(y_frag.x[0]), "=f"(y_frag.x[1]), "=f"(y_frag.x[2]), "=f"(y_frag.x[3]), "=f"(y_frag.x[4]), "=f"(y_frag.x[5]), "=f"(y_frag.x[6]), "=f"(y_frag.x[7])
                     :  "r"(X[1]),  "r"(X[3]),  "r"(X[5]),  "r"(X[7]),  "r"(X[9]),  "r"(X[11]), "r"(X[13]), "r"(X[15]), \
                        "r"(W[1]),  "r"(W[3]),  "r"(W[5]),  "r"(W[7]),  "r"(W[9]),  "r"(W[11]), "r"(W[13]), "r"(W[15]), \
                       "f"(y_frag.x[0]), "f"(y_frag.x[1]), "f"(y_frag.x[2]), "f"(y_frag.x[3]), "f"(y_frag.x[4]), "f"(y_frag.x[5]), "f"(y_frag.x[6]), "f"(y_frag.x[7]));
}




__inline__ __device__ void wmma16x16x16(float *O,
                                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> &x_frag,
                                        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> &w_frag)
{                       
        const __half *X = reinterpret_cast<const __half *>(&x_frag.x);
        const __half *W = reinterpret_cast<const __half *>(&w_frag.x);

        asm volatile("wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32"
                     "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
                     "{%8,  %9,  %10, %11, %12, %13, %14, %15},"
                     "{%16, %17, %18, %19, %20, %21, %22, %23},"
                     "{%24, %25, %26, %27, %28, %29, %30, %31};"
                     : "=f"(O[0]), "=f"(O[1]), "=f"(O[2]), "=f"(O[3]), "=f"(O[4]), "=f"(O[5]), "=f"(O[6]), "=f"(O[7])
                     :  "r"(X[0]),  "r"(X[2]),  "r"(X[4]),  "r"(X[6]),  "r"(X[8]),  "r"(X[10]), "r"(X[12]), "r"(X[14]), \
                        "r"(W[0]),  "r"(W[2]),  "r"(W[4]),  "r"(W[6]),  "r"(W[8]),  "r"(W[10]), "r"(W[12]), "r"(W[14]), \
                        "f"(O[0]),  "f"(O[1]),  "f"(O[2]),  "f"(O[3]),  "f"(O[4]),  "f"(O[5]),  "f"(O[6]),  "f"(O[7]));

                        
        asm volatile("wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32"
                     "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
                     "{%8,  %9,  %10, %11, %12, %13, %14, %15},"
                     "{%16, %17, %18, %19, %20, %21, %22, %23},"
                     "{%24, %25, %26, %27, %28, %29, %30, %31};"
                     : "=f"(O[0]), "=f"(O[1]), "=f"(O[2]), "=f"(O[3]), "=f"(O[4]), "=f"(O[5]), "=f"(O[6]), "=f"(O[7])
                     :  "r"(X[1]),  "r"(X[3]),  "r"(X[5]),  "r"(X[7]),  "r"(X[9]),  "r"(X[11]), "r"(X[13]), "r"(X[15]), \
                        "r"(W[1]),  "r"(W[3]),  "r"(W[5]),  "r"(W[7]),  "r"(W[9]),  "r"(W[11]), "r"(W[13]), "r"(W[15]), \
                        "f"(O[0]),  "f"(O[1]),  "f"(O[2]),  "f"(O[3]),  "f"(O[4]),  "f"(O[5]),  "f"(O[6]),  "f"(O[7]));
}



__inline__ __device__ void frag_to_mem(float *frag, float *mem, int ld)
{
  const auto func_y = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {
                
        
    #pragma unroll
        for (unsigned f = 0; f < fragment_index_count; f++)
          mem[i*ld + j] = frag[frag_index_list[f]];
    };

  __syncwarp();
  wmma_foreach_ij_acc(func_y);
}







__device__ __inline__ void gmem_to_smem_xor(const float *gmem_ptr, float &smem, const int trunc)
{
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem);
  
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
                :: "r"(smem_int_ptr),
                   "l"(gmem_ptr),
                   "n"(16),
                   "r"(trunc)); // incorrect 0 padding yet
}





template<int WMMA_T, int wk>
__global__ void wmma_cp_async_blocking(const float *x, const float *w,
                      float *out, const int B, const int C, const int OC,
                      const int bx, const int by,
                      const int wx, const int wy, const int wx_per_bx, const int wy_per_by,
                      const int X_WARPS, const int Y_WARPS) {


  int laneId = ( threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
  int mw = laneId / 4;
  int ml = laneId % 4;

  int warp_y = threadIdx.y;
  int warp_x = (threadIdx.x / 32);

  int block_x = blockIdx.x;
  int block_y = blockIdx.y;




  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag;
  // wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag_delta;
  // wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag_delta;
  // wmma::fragment<wmma::accumulator, 16, 16, 16, float> y_frag;


  
  const int wy_count = 2;
  const int wy_loop = 2;

  float   acc_frag[4*2*8];
  // float delta_frag[4*2*8];

  for (int i=0; i<4*2*8; ++i)
  {
    acc_frag[i] = 0.0f;
    // delta_frag[i] = 0.0f;
  }





  extern __shared__ float smem[];
  float *out_smem = smem;
  // __half *hsmem = reinterpret_cast<__half*>(smem + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T));

  float *x_smem     = smem;
  float *w_smem     = smem + by*wk;
  
  
  int xor_addr = smem_xor_cp_async(laneId);

#pragma unroll
  for (int tile=0; tile<C; tile+=wk)
  {
    // warp * mw_size * i_size + mw*i_size + i
    
    int row_aux1 = warp_x*4 + ml;
    








    if (row_aux1<WMMA_T)
    {
      int row = block_y*by + warp_y*WMMA_T + row_aux1;
      float const *gmem_ptr = x + row*C + tile+mw*4;

      // extra *2 to accomodate 32 instead of 16 C (i.e, the whole warpSize)
      //       *4 is necessary as it needs to accomodate 4 consecutive floats
      gmem_to_smem_xor(gmem_ptr,  *(x_smem + (warp_y*WMMA_T+ warp_x*4)*wk + xor_addr),  (row<B) ? std::min((( C-(tile+mw*4)) /4)*4, 16) : 0);
    }
    





    for (int aux_i=0; aux_i<wy_loop; ++aux_i)
    {
      int row_aux2 = (aux_i*wy_count+warp_y)*4 + ml;

      if (row_aux2<WMMA_T)
      {
        int row = block_x*bx + warp_x*WMMA_T + row_aux2;
        float const *gmem_ptr = w + row*C + tile+mw*4;
        
        gmem_to_smem_xor(gmem_ptr,  *(w_smem + (warp_x*WMMA_T + (aux_i*wy_count + warp_y)*4)*wk + xor_addr),  (row<OC) ? std::min((( C-(tile+mw*4)) /4)*4, 16) : 0);        
      }
    }



    asm volatile("cp.async.wait_all;");
    __syncthreads();




    

    // if ((warp_x+warp_y+laneId+block_x+block_y)==0)
    // {
    //   printf("blocking y: %d\n", y_blocking);
    //   for (int i=0; i<4*16; ++i)
    //   {
    //     printf("%.2f, ", x_smem[i+y_blocking*wy*wk]);
    //     if((i+1)%16==0)
    //       printf("\n");
    //   }
    //   printf("---------------------\n\n");
    // }

    if ((block_y*by + warp_y*WMMA_T)<B && (block_x*bx + warp_x*WMMA_T)<OC)
    {
      __syncwarp();
      for (int k_stride=0; k_stride<2; ++k_stride)
      {
        
        smem_xor_to_reg_A(x_frag, x_smem + (warp_y*WMMA_T)*wk, wk, k_stride);
        smem_xor_to_reg_B(w_frag, w_smem + (warp_x*WMMA_T)*wk, wk, k_stride);
        // smem_xor_to_reg_A_ec(x_frag, x_frag_delta, x_smem + (y_blocking*wy + warp_y*WMMA_T)*wk, wk, k_stride);
        // smem_xor_to_reg_B_ec(w_frag, w_frag_delta, w_smem + (x_blocking*wx + warp_x*WMMA_T)*wk, wk, k_stride);
        

        wmma16x16x16(acc_frag, x_frag, w_frag);
        // wmma16x16x16(delta_frag+(x_blocking*wy_per_by+y_blocking)*8, x_frag_delta, w_frag);
        // wmma16x16x16(delta_frag+(x_blocking*wy_per_by+y_blocking)*8, x_frag, w_frag_delta);

        // wmma::mma_sync(y_frag, x_frag, w_frag, y_frag);

        // if ((warp_x+warp_y+laneId+block_x+block_y)==0)
        // {
        //   printf("blocking x: %d, y: %d -- k stride: %d\n", x_blocking, y_blocking, k_stride);
        //   for (int i=0; i<8; ++i)
        //   {
        //     printf("%.2f, ", acc_frag[i+(x_blocking*wy_per_by+y_blocking)*8]);
        //   }
        //   printf("----------\n\n");
        // }

      }
    }

  }
  

  



  float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;

  

  __syncthreads();

  int threaded_row = block_y*by + warp_y*WMMA_T;
  int threaded_col = block_x*bx + warp_x*WMMA_T;

  if (threaded_row<B && threaded_col<OC && (warp_y*WMMA_T)<B && (warp_x*WMMA_T)<OC)
  {
    
    
    frag_to_mem(acc_frag, _out, X_WARPS*WMMA_T);
    // wmma::store_matrix_sync(_out, y_frag, X_WARPS*WMMA_T, wmma::mem_row_major);
    // frag_to_mem_ec(acc_frag+(x_blocking*wy_per_by+y_blocking)*8, delta_frag+x_blocking*2+y_blocking, _out, X_WARPS*WMMA_T);
    
    

#pragma unroll
    for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)(warpSize)); ++tile)
    {
      int tile_idx = tile*warpSize + laneId;

      int row =  tile_idx / WMMA_T;
      int col = (tile_idx % WMMA_T);


      if((threaded_row+row)<B  &&  (threaded_col+col)<OC && row<WMMA_T)
        out[(threaded_row+row)*OC + threaded_col+col] = _out[row*(X_WARPS*WMMA_T)+col];

    }
  }
}





























template<int WMMA_T, int X_WARPS, int Y_WARPS>
__global__ void wmma_dx_cp_async(const float *dx, const float *w,
                      float *dy, const int B, const int C, const int OC) {

  int laneId = ( threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
  int mw = laneId / 4;
  int ml = laneId % 4;

  int warp_y = threadIdx.y;
  int warp_x = (threadIdx.x / 32);


  const uint32_t warpX{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};  // C
  const uint32_t warpY{blockIdx.y * blockDim.y + threadIdx.y};               // B

  // warpX = (oc*X_WARPS + warp_x)




  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> y_frag;

  



  wmma::fill_fragment(y_frag, 0.0f);


  extern __shared__ float smem[];
  float *out_smem = smem;
  

  float *hsmem = smem + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T);

  float *x_smem     = hsmem;
  float *w_smem     = hsmem + Y_WARPS*WMMA_T*(WMMA_T*2);
  
  
  int k_count=0;
  int k_stride;

#pragma unroll
  for (int tile=0; tile<C; tile+=WMMA_T)
  {
    // warp * mw_size * i_size + mw*i_size + i
    
    k_stride = k_count % 2;
    k_count++;


    // each block deals with a 64x16 tile
    

    int row_aux1 = warp_x*4 + ml;
    int row_aux2 = warp_y*4 + ml;
    
    int xor_addr = smem_xor_cp_async(laneId);
    
    if (k_stride==0) {



      if (row_aux1<WMMA_T)
      {
        float const *gmem_ptr = &dy[(warpY*WMMA_T+row_aux1)*C + tile+mw*4];
        
        // extra 2 to accomodate 32 instead of 16 C
        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&x_smem[(warp_y*WMMA_T+ warp_x*4)*WMMA_T*2 + xor_addr]);
        
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
                      :: "r"(smem_int_ptr),
                        "l"(gmem_ptr),
                        "n"(16),
                        "r"(((warpY*WMMA_T+row_aux1)<B) ? std::min(((C-tile+mw*4)/4)*4, 16) : 0)); // incorrect 0 padding yet
                        // "r"(((warpY*WMMA_T+row_aux1)<B) ? 16 : 0)); // incorrect 0 padding yet
        
      }

      if (row_aux2<WMMA_T)
      {
        // w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = *(w + (warpX*WMMA_T+row_aux2)*C + tile+ml);
        
        float const *gmem_ptr = w + (warpX*WMMA_T+row_aux2)*C + tile+mw*4;
        
        // extra 2 to accomodate 32 instead of 16 C
        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&w_smem[(warp_x*WMMA_T+ warp_y*4)*WMMA_T*2 + xor_addr]);
        
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
                      :: "r"(smem_int_ptr),
                        "l"(gmem_ptr),
                        "n"(16),
                        "r"(((warpX*WMMA_T+row_aux2)<OC) ? std::min(((C-tile+mw*4)/4)*4, 16) : 0));
                        // "r"(((warpX*WMMA_T+row_aux2)<OC) ? 16 : 0));
        
      }
    }
    // asm volatile("cp.async.commit_group;\n" ::);
    
    asm volatile("cp.async.wait_all;");
    
    __syncthreads();



    if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC)
    {

    
    const auto func_x = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {


        int wi = i/4;
        int xi = i%4;

        int xj = j/4;
        int wj = j%4;

        int offset = smem_dexor_from_cp_async(xi, xj*2 + k_stride)+wj;

      
        __half tmp = __float2half(*(x_smem + (warp_y*WMMA_T+ wi*4)*WMMA_T*2 + offset));


#pragma unroll
        for (unsigned f = 0; f < fragment_index_count; f++)
            x_frag.x[frag_index_list[f]] = tmp;
    };
    
    __syncwarp();
    wmma_foreach_ij(
        x_frag,
        func_x
      );


    
    const auto func_w = [&](const unsigned* frag_index_list,
          const unsigned fragment_index_count,
          const unsigned i,
          const unsigned j) {
        

          int wj = j/4;
          int xj = j%4;
        
          int xi = i/4;
          int wi = i%4;


          int offset = smem_dexor_from_cp_async(xj, xi*2+k_stride)+wi;

        
          __half tmp = __float2half(*(w_smem + (warp_x*WMMA_T+wj*4)*WMMA_T*2 + offset));
  #pragma unroll
          for (unsigned f = 0; f < fragment_index_count; f++)
            w_frag.x[frag_index_list[f]] = tmp;
      };

    __syncwarp();
    wmma_foreach_ij(
        w_frag,
        func_w
      );


      wmma::mma_sync(y_frag, x_frag, w_frag, y_frag);

    
    }
    
  }


  if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC && (warp_y*WMMA_T)<B && (warp_x*WMMA_T)<OC)
  { 

    float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;
    wmma::store_matrix_sync(_out, y_frag, X_WARPS*WMMA_T, wmma::mem_row_major);

    
#pragma unroll
    for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)(warpSize)); ++tile)
    {
      int tile_idx = tile*warpSize + laneId;

      int row =  tile_idx / WMMA_T;
      int col = (tile_idx % WMMA_T);


      if((warpY*WMMA_T+row)<B  &&  (warpX*WMMA_T+col)<OC && row<WMMA_T)
        dy[(warpY*WMMA_T+row)*OC + warpX*WMMA_T+col] = _out[row*(X_WARPS*WMMA_T)+col];

    }
  }
}






























template<int WMMA_T, int X_WARPS, int Y_WARPS>
__global__ void wmma_pingpong(const float *x, const float *w,
                      float *out, const int B, const int C, const int OC) {

  int laneId = ( threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
  int mw = laneId / WMMA_T;
  int ml = laneId % WMMA_T;

  int warp_y = threadIdx.y;
  int warp_x = (threadIdx.x / warpSize);

  int s=2;
  int circular_smem_counter=0;


  uint32_t warpX;                                                     // OC
  uint32_t warpY{blockIdx.y * blockDim.y + threadIdx.y};               // B

  // warpX = (oc*X_WARPS + warp_x)





  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> y_frag;

  //using FRAG_T = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;

  wmma::fill_fragment(y_frag, 0.0f);


  extern __shared__ float smem[];
  float *out_smem = smem;
  __half *hsmem = reinterpret_cast<__half*>(smem + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T));

  __half *x_smem_base  = hsmem;
  __half *w_smem_base  = hsmem + (Y_WARPS*WMMA_T)*WMMA_T;
  
  __half *x_smem, *w_smem;
  





  if (warp_x>=4)
  {
    warp_x-=4;
    warpX = (blockIdx.x*(blockDim.x/2))/warpSize + warp_x;
    
    
  
    for (int tile=0; tile<C; tile+=WMMA_T)
    {

      int tgt_smem = circular_smem_counter % s;


      x_smem = x_smem_base + tgt_smem*((Y_WARPS+X_WARPS)*WMMA_T*WMMA_T);
      w_smem = w_smem_base + tgt_smem*((Y_WARPS+X_WARPS)*WMMA_T*WMMA_T);

      

  
      for (int i=0; i<2; ++i)
      {
        // warp*mw_size*i_size + mw*i_size + i
        int row_aux1 = warp_x*((int)(warpSize/WMMA_T))*2 + mw*2+i;
        int row_aux2 = warp_y*((int)(warpSize/WMMA_T))*2 + mw*2+i;
        
        if (row_aux1<WMMA_T)
        {
          if ((warpY*WMMA_T+row_aux1)<B && (tile+ml)<C)
            x_smem[(warp_y*WMMA_T+row_aux1)*WMMA_T + ml] = __float2half(*(x + (warpY*WMMA_T+row_aux1)*C + tile+ml));
          else
            x_smem[(warp_y*WMMA_T+row_aux1)*WMMA_T + ml] = 0;
        }

        if (row_aux2<WMMA_T)
        {
          if ((warpX*WMMA_T+row_aux2)<OC && (tile+ml)<C)
            w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = __float2half(*(w + (warpX*WMMA_T+row_aux2)*C + tile+ml));
          else
            w_smem[(warp_x*WMMA_T+row_aux2)*WMMA_T + ml] = 0;
        }
      }
      

      
      asm volatile("bar.sync 0, 1024;"); // producer waits consumer
      
      asm volatile("bar.arrive 1, 1024;"); // producer ends


      circular_smem_counter++;
      
      

      // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
      //   printf("Producer finished, tile: %d/%d.\n", tile, C);




      
      //asm volatile("bar.sync 2, 512;");
      // __syncthreads();

    }

    // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
    //   printf("Producer exits.\n");

    return; // return is a must, otherwise the if below bugs
  }
  else if (warp_x<4)
  {
    warpX = (blockIdx.x*(blockDim.x/2))/warpSize + warp_x;


    asm volatile("bar.arrive 0, 1024;");

  
    for (int tile=0; tile<C; tile+=WMMA_T)
    {


      int tgt_smem = circular_smem_counter % s;

      x_smem = x_smem_base + tgt_smem*((Y_WARPS+X_WARPS)*WMMA_T*WMMA_T);
      w_smem = w_smem_base + tgt_smem*((Y_WARPS+X_WARPS)*WMMA_T*WMMA_T);


      


      // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
      //   printf("\t\t\t\t\tConsumer wait %d.\n", tile);

      asm volatile("bar.sync 1, 1024;"); // consumer waits producer

      // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
      //   printf("\t\t\t\t\tConsumer go %d.\n", tile);




      if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC)
      {


        wmma::load_matrix_sync(x_frag, x_smem+warp_y*WMMA_T*WMMA_T, WMMA_T);
        wmma::load_matrix_sync(w_frag, w_smem+warp_x*WMMA_T*WMMA_T, WMMA_T);



        wmma::mma_sync(y_frag, x_frag, w_frag, y_frag);


      }
      

      asm volatile("bar.arrive 0, 1024;"); // consumer ends

      // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
      //   printf("\t\t\t\t\tConsumer finished, tile: %d.\n", tile);
      
      
      circular_smem_counter++;
    }


    // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
    //   printf("\t\tConsumer exits.\n");


    if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC && (warp_y*WMMA_T)<B && (warp_x*WMMA_T)<OC)
    { 
      // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
      //   printf("\t\t ------ NOW STORE OUTPUT ------\n");

      float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;
      wmma::store_matrix_sync(_out, y_frag, X_WARPS*WMMA_T, wmma::mem_row_major);

      // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
      //   printf("\t\t ------ post wmma store ------\n");


      
      
  
      for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)warpSize); ++tile)
      {
        int tile_idx = tile*warpSize + laneId;

        int row = tile_idx / WMMA_T;
        int col = tile_idx % WMMA_T;
        


        // if ((blockIdx.y+ warp_y+laneId)==0)
        //   printf("warpX: %d\t warpX offset: %d\t OC: %d\n", warpX, warpX*WMMA_T, OC);
        // if ((blockIdx.x+ warp_x+laneId)==0)
        //   printf("warpY: %d, out: %f\n", warpY);

        if((warpY*WMMA_T+row)<B  &&  (warpX*WMMA_T+col)<OC  &&  row<WMMA_T)
          out[(warpY*WMMA_T+row)*OC + warpX*WMMA_T+col] = _out[row*(X_WARPS*WMMA_T)+col];

      }
    }
    // if ((blockIdx.x+blockIdx.y+warp_x+warp_y+laneId)==0)
    //     printf("\t\t ------ post tiled ------\n");
    
  }
}
















// template<int WMMA_T, int X_WARPS, int Y_WARPS>
// __global__ void wmma_cutlass(const float *x, const float *w,
//                       float *out, const int B, const int C, const int OC) {

//   int tid = threadIdx.y * blockDim.x + threadIdx.x;
//   int laneId = tid % warpSize;
//   int mw = laneId / WMMA_T;
//   int ml = laneId % WMMA_T;

//   int warp_y = threadIdx.y;
//   int warp_x = (threadIdx.x / 32);


//   const uint32_t warpX{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};  // OC
//   const uint32_t warpY{blockIdx.y * blockDim.y + threadIdx.y};               // B

//   // warpX = (oc*X_WARPS + warp_x)


//   using ColumnMajor = cutlass::layout::ColumnMajor;
//   using RowMajor = cutlass::layout::RowMajor;

//   using Mma = cutlass::gemm::warp::DefaultMmaTensorOp<
//     cutlass::gemm::GemmShape<16,16,16>,
//     cutlass::gemm::GemmShape<16,16,16>,
//     cute::half_t, RowMajor,
//     cute::half_t, ColumnMajor,
//     float, RowMajor
//   >;


//   extern __shared__ float smem[];
//   float *out_smem = smem;
//   __half *hsmem = reinterpret_cast<__half*>(smem + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T));

//   __half *x_smem     = hsmem;
//   __half *w_smem     = hsmem + Y_WARPS*WMMA_T*(WMMA_T);

//   typename Mma::IteratorA iter_A({x_smem, C}, tid);
//   typename Mma::IteratorB iter_B({w_smem, C}, tid);
//   typename Mma::IteratorC iter_C({out_smem, OC}, tid);

//   typename Mma::FragmentA x_frag;
//   typename Mma::FragmentB w_frag;
//   typename Mma::FragmentC y_frag;
  
//   Mma mma;
  
//   y_frag.clear();

// #pragma unroll
//   for (int tile=0; tile<C; tile+=WMMA_T)
//   {
//     iter_A.load(x_frag);
//     iter_B.load(w_frag);

//     ++iter_A, ++iter_B;
    
//     mma(y_frag, x_frag, w_frag, y_frag);
//   }


//   if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC && (warp_y*WMMA_T)<B && (warp_x*WMMA_T)<OC)
//   { 

//     // float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;
//     // wmma::store_matrix_sync(_out, y_frag, X_WARPS*WMMA_T, wmma::mem_row_major);
//     iter_C.store(y_frag);

    
// #pragma unroll
//     for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)warpSize); ++tile)
//     {
//       int tile_idx = tile*warpSize + laneId;

//       int row = tile_idx / WMMA_T;
//       int col = tile_idx % WMMA_T;


//       if((warpY*WMMA_T+row)<B  &&  (warpX*WMMA_T+col)<OC && row<WMMA_T)
//         out[(warpY*WMMA_T+row)*OC + warpX*WMMA_T+col] = _out[row*(X_WARPS*WMMA_T)+col];

//     }
//   }
// }




















template<int WMMA_T, int X_WARPS, int Y_WARPS>
__global__ void wmma_mult_kernel_(const float *x, const float *w,
                      float *out, const int B, const int C, const int OC) {

  int laneId = ( threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
  int mw = laneId / WMMA_T;
  int ml = laneId % WMMA_T;

  int warp_y = threadIdx.y;
  int warp_x = (threadIdx.x / 32);


  const uint32_t warpX{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};  // OC
  const uint32_t warpY{blockIdx.y * blockDim.y + threadIdx.y};               // B

  // warpX = (oc*X_WARPS + warp_x)




  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> y_frag;

  using FRAG_T = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;



  wmma::fill_fragment(y_frag, 0.0f);


  extern __shared__ float smem[];
  float *out_smem = smem;
  

  for (int tile=0; tile<C; tile+=WMMA_T)
  {
    

    
    wmma::fill_fragment(x_frag, 0.0f);
    const auto func_x = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {

      if((warpY*WMMA_T+i)<B && (tile+j)<C)
      {
        __half tmp = __float2half(*(x + (warpY*WMMA_T+i)*C + tile + j));
#pragma unroll
        for (unsigned f = 0; f < fragment_index_count; f++)
            x_frag.x[frag_index_list[f]] = tmp;
      } // else did not work, so fill_fragment is a workaround
    };
    
    __syncwarp();
    wmma_foreach_ij(
        x_frag,
        func_x
      );



    wmma::fill_fragment(w_frag, 0.0f);
    const auto func_w = [&](const unsigned* frag_index_list,
          const unsigned fragment_index_count,
          const unsigned i,
          const unsigned j) {
        
        if((warpX*WMMA_T+j)<OC && (tile+i)<C)
        { 
          __half tmp = __float2half(*(w + (warpX*WMMA_T+j)*C + tile+i));
  #pragma unroll
          for (unsigned f = 0; f < fragment_index_count; f++)
            w_frag.x[frag_index_list[f]] = tmp;
        }
      };

    __syncwarp();
    wmma_foreach_ij(
        w_frag,
        func_w
      );




    __syncwarp();
    if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC)
      wmma::mma_sync(y_frag, x_frag, w_frag, y_frag);

    
  }


  if ((warpY*WMMA_T)<B && (warpX*WMMA_T)<OC && (warp_y*WMMA_T)<B && (warp_x*WMMA_T)<OC)
  { 

    float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;
    wmma::store_matrix_sync(_out, y_frag, X_WARPS*WMMA_T, wmma::mem_row_major);

    __syncthreads();
    
#pragma unroll
    for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)warpSize); ++tile)
    {
      int tile_idx = tile*warpSize + laneId;

      int row = tile_idx / WMMA_T;
      int col = tile_idx % WMMA_T;

      if((warpY*WMMA_T+row)<B  &&  (warpX*WMMA_T+col)<OC && row<WMMA_T)
        out[(warpY*WMMA_T+row)*OC + warpX*WMMA_T+col] = _out[row*(X_WARPS*WMMA_T)+col];

    }
  }
}








__global__ void to_half(__half *y, const float *x, int dims_prod)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if(idx>=dims_prod)
    return;

  y[idx] = __float2half(x[idx]);

}

void matmul_forward(float* out,
                     float* inp, float* W,
                     int B, int C, int OC, int thread_id) {
        
  const float alpha = 1.0f;
  const float beta = 0.0f;
  

  //std::cout << "matmul forward. B: " << B << " C: " << C << " OC: " << OC << "\n";


  if (thread_id==0)
  {
    //cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B, C, &alpha, W, C, inp, C, &beta, out, OC));
    
    cudaStream_t stream = ThreadsStream[thread_id];


    
    /*
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(std::ceil(OC/(float)TILE_SIZE), std::ceil(B/(float)TILE_SIZE));
    int shared_mem_size = 2*TILE_SIZE*TILE_SIZE*sizeof(float);
    mult_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(inp, W, out, TILE_SIZE, TILE_SIZE*TILE_SIZE, B, C, OC);
    */
    
    constexpr int num_warps_x{4};
    constexpr int num_warps_y{4};
    

    constexpr int WMMA_T{16};
    dim3 block_size(num_warps_x * WARP_SIZE, num_warps_y);
    dim3 grid_size(std::ceil((OC + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((B + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));

    int shared_mem_size = num_warps_y*WMMA_T*WMMA_T*num_warps_x*sizeof(float);


    // float *bank;
    // cudaMalloc(&bank, 32*16*sizeof(float));
    
    // set_to_one_kernel<<<16, 32,0,stream>>>(bank, 16*32);
    
    wmma_mult_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size, block_size, shared_mem_size, stream>>>(inp, W, out, B, C, OC);


    //PrintTensorF(bank, 32, 16);
    
  }
  else
  {
    cudaStream_t stream = ThreadsStream[thread_id];

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(std::ceil(OC/(float)TILE_SIZE), std::ceil(B/(float)TILE_SIZE));
    int shared_mem_size = 2*TILE_SIZE*TILE_SIZE*sizeof(float);

    mult_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(inp, W, out, TILE_SIZE, TILE_SIZE*TILE_SIZE, B, C, OC);
  }
  
  
  

  /*
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor = cutlass::layout::RowMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,
                                                  RowMajor,
                                                  float,
                                                  ColumnMajor,
                                                  float,
                                                  RowMajor>;
  CutlassGemm gemm_operator;

  CutlassGemm::Arguments args({B, OC, C},
                              {inp, C},
                              {weight, C},
                              {out, OC},
                              {out, OC},
                              {alpha, beta});
                              
  gemm_operator(main_stream);
  gemm_operator(args);
  */
    
  /* //bias
  if (bias != NULL) {
      int block_size = sqrt_block_size * sqrt_block_size;
      int grid_size = ceil_div(OC * B * T, block_size);
      add_bias<<<grid_size, block_size>>>(out, bias, B, T, OC);
  }
  */
}






extern "C" Tensor *CudaMult(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w, int thread_id) {

  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;


  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(Ldims);
  int input_dims_prod = DimsProd(linear_layer_dims);
  
  int resultingDimsProd = resultingDimsProdOnMult(linear_layer_dims, Rdims);




  float* device_y = get_from_pool(thread_id, resultingDimsProd, "cuda mult");
  
  

  if (Ldims.size()<2)
    LogErrorS("Tensors multiplication requires at least 2 dimensions.");



  tensor_x->Sync();
  tensor_w->Sync();

  matmul_forward(device_y, device_x, device_w,
                  linear_layer_dims[0], linear_layer_dims[1],
                  Rdims[0], thread_id);

  
  
  std::vector<float> new_dims = NewDimsOnMult(Ldims, Rdims);

  Tensor *new_tensor = createTensor(device_y, new_dims, resultingDimsProd, false, "");
  new_tensor->AttrNodes(tensor_x, tensor_w, mult_op);
  return new_tensor;
}




__global__ void btc_mult_kernel(float *out, const float *x, const float *w, const int B, const int Tx, const int Tw, const int C, const int tile_size, const int tile_offset)
{

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * blockDim.y + threadIdx.y; // [B, T]
  int col = blockIdx.x * blockDim.x + threadIdx.x; // [T]


  //int b = row / Tx;
  //int t = row % Tx;
  int t = row;


  extern __shared__ float smem_x[];
  float *smem_w = smem_x + tile_offset;



  for (int b=0; b<B; ++b)
  {
    float y=0;

    for (int i=0; i < ceilf(C/(float)tile_size); ++i)
    {
      int _col1 = i*tile_size + tx;
      int _col2 = i*tile_size + ty;

      if (b<B && t<Tx && _col1<C)
        smem_x[tx*tile_size + ty] = x[(b*Tx + t)*C + _col1];
      else
        smem_x[tx*tile_size + ty] = 0;


      if (b<B && col<Tw && _col2<C)
        smem_w[ty*tile_size + tx] = w[(b*Tx + col)*C + _col2];
      else
        smem_w[ty*tile_size + tx] = 0;

      __syncthreads();

      for(int j=0; j<tile_size; ++j)
        y += smem_x[j* tile_size + ty] * smem_w[j* tile_size + tx];
      
      __syncthreads();
    }


    // [B, T, T]
    if(b<B && t<Tx && col<Tw)
      out[(b*Tx + t)*Tw + col] = y;
  }
}



extern "C" Tensor *btc_mult(int thread_id, Tensor *x, Tensor*w)
{

  std::vector<float> Ldims, Rdims, new_dims;
  Ldims = x->dims;
  Rdims = w->dims;
  float *device_x = x->tensor_ptr;
  float *device_w = w->tensor_ptr;

  

  int B, Tx, C, Tw;

  B  = Ldims[0];
  Tx = Ldims[1];
  C  = Ldims[2];

  Tw = Rdims[1];


  new_dims = {(float)B, (float)Tx, (float)Tw};
  int new_dims_prod = DimsProd(new_dims);


  float* device_y = get_from_pool(thread_id, new_dims_prod, "btc mult");

  x->Sync();
  w->Sync();



  dim3 block_size(TILE_SIZE, TILE_SIZE);
  dim3 grid_size(std::ceil(Tw/(float)TILE_SIZE), std::ceil((Tx)/(float)TILE_SIZE));
  int shared_mem_size = 2*TILE_SIZE*TILE_SIZE*sizeof(float);

  cudaStream_t stream = ThreadsStream[thread_id];


  
  btc_mult_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(device_y, device_x, device_w, B, Tx, Tw, C, TILE_SIZE, TILE_SIZE_SQ);
  //mult_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(device_x, device_w, device_y, TILE_SIZE, TILE_SIZE*TILE_SIZE, Tw, Tx, C);
  

  Tensor *new_tensor = createTensor(device_y, new_dims, new_dims_prod, false, "");
  new_tensor->AttrNodes(x, w, mult_op);
  return new_tensor;
}






__global__ void btc_mult_kernelT(float *out, const float *x, const float *w, const int B, const int Tx, const int Tw, const int C, const int tile_size, const int tile_offset)
{

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * blockDim.y + threadIdx.y; // [Tx]
  int col = blockIdx.x * blockDim.x + threadIdx.x; // [C]



  //int b = row / Tx;
  //int t = row % Tx;


  extern __shared__ float smem_x[];
  float *smem_w = smem_x + tile_offset;




  
  // [B, Tx, Tw], consider T as Tw, row as Tx

  for (int b=0; b<B; ++b)
  {
    float y=0;
    __syncthreads();

    for (int i=0; i < ceilf(Tw/(float)tile_size); ++i)
    {
      int _col1 = i*tile_size + tx;
      int _col2 = i*tile_size + ty;

      if (b<B && row<Tx && _col1<Tw)
        smem_x[tx*tile_size + ty] = x[b*Tx*Tw + row*Tw + _col1];
      else
        smem_x[tx*tile_size + ty] = 0;


      if (b<B && _col2<Tw && col<C)
        smem_w[ty*tile_size + tx] = w[b*Tw*C + _col2*C + col];
      else
        smem_w[ty*tile_size + tx] = 0;

      __syncthreads();

      for(int j=0; j<tile_size; ++j)
        y += smem_x[j*tile_size + ty] * smem_w[j*tile_size + tx];
      
      __syncthreads();
    }


    // [B, Tx, C]
    if (b<B && row<Tx && col<C)
      out[b*Tx*C + row*C + col] = y;
      
  }
}



extern "C" Tensor *btc_multT(int thread_id, Tensor *x, Tensor*w)
{


  std::vector<float> Ldims, Rdims, new_dims;
  Ldims = x->dims;
  Rdims = w->dims;
  float *device_x = x->tensor_ptr;
  float *device_w = w->tensor_ptr;

  

  int B, Tx, Tw, C;

  B  = Ldims[0];
  Tx = Ldims[1];
  Tw = Ldims[2];

  C  = Rdims[2];


  new_dims = {(float)B, (float)Tx, (float)C};
  int new_dims_prod = DimsProd(new_dims);


  float* device_y = get_from_pool(thread_id, new_dims_prod, "cuda mult");



  dim3 block_size(TILE_SIZE, TILE_SIZE);
  dim3 grid_size(std::ceil(C/(float)TILE_SIZE), std::ceil(Tx/(float)TILE_SIZE));
  int shared_mem_size = 2*TILE_SIZE_SQ*sizeof(float);

  cudaStream_t stream = ThreadsStream[thread_id];
  
  btc_mult_kernelT<<<grid_size, block_size, shared_mem_size, stream>>>(device_y, device_x, device_w, B, Tx, Tw, C, TILE_SIZE, TILE_SIZE_SQ);


  Tensor *new_tensor = createTensor(device_y, new_dims, new_dims_prod, false, "");
  new_tensor->AttrNodes(x, w, mult_op);
  return new_tensor;
}





__global__ void add_forward(float *y, const float *x,
                            const float *w, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = x[i] + w[i];
}
__global__ void add_inplace(float *y, const float *x,
                                    int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = y[i] + x[i];
}

__global__ void sub_forward(float *y, const float *x,
                            const float *w, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = x[i] - w[i];
}


__global__ void equal_forward(float *y, const float *x,
                            const float *w, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = (x[i]==w[i]) ? 1.0f : 0.0f;
}


__global__ void broadcast_lastdim_add(float *y, const float *x,
                            const float *w, int dims_prod, int C) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int w_id = tid / C;

    if (tid >= dims_prod)
      return;



    y[tid] = x[tid] + w[w_id];
}

extern "C" Tensor *CudaAdd(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w, int thread_id) {

  //std::cout << "Cuda add of\n      L " << tensor_x.name << "  &  R " << tensor_w.name << "\n";
    
  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;




  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(Ldims);
  float dims_prod = tensor_x->dims_prod;


  float* device_y = get_from_pool(thread_id, dims_prod, "add");


  tensor_x->Sync();
  tensor_w->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];



  if (Ldims==Rdims)
  {

    int grid_size, block_size;
    std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];
    
    add_forward<<<grid_size, block_size, 0, stream>>>(device_y, device_x, device_w, dims_prod);
    

    Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");
    new_tensor->AttrNodes(tensor_x, tensor_w, add_op);
    return new_tensor;
  }

  

  
  if(RemoveLastDim(Ldims)==Rdims||(RemoveLastDim(Ldims)==RemoveLastDim(Rdims)&&Rdims[Rdims.size()-1]==1))
  {
    int grid_size, block_size;
    std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];
    
    broadcast_lastdim_add<<<grid_size, block_size, 0, stream>>>(device_y, device_x, device_w, dims_prod, tensor_x->dims[tensor_x->dims.size()-1]);
    

    Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");
    new_tensor->AttrNodes(tensor_x, tensor_w, broadcast_lastdim_add_op);
    return new_tensor;
  }


  if (Ldims!=Rdims)
  {
    LogErrorS("Tried to add tensors of different dimenstions.");
    std::cout << "   Left tensor dims " << "\n   ";
    PrintDims(Ldims);
    std::cout << "\n   Right tensor dims " << "\n   ";
    PrintDims(Rdims);
    std::cout << "\n\n";
    return nullptr;
  }
}


extern "C" Tensor *CudaSub(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w, int thread_id) {

  //std::cout << "Cuda add of\n      L " << tensor_x.name << "  &  R " << tensor_w.name << "\n";
    
  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;


  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(Ldims);
  float dims_prod = tensor_x->dims_prod;




  float* device_y = get_from_pool(thread_id, dims_prod,"sub");



  int grid_size = dims_prod;
  int block_size = 512;
  
  tensor_x->Sync();
  tensor_w->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  sub_forward<<<grid_size, block_size, 0, stream>>>(device_y, device_x, device_w, dims_prod);
  
  
  Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");  
  new_tensor->AttrNodes(tensor_x, tensor_w, sub_op);
  return new_tensor;
}


extern "C" Tensor *CudaEqual(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w, int thread_id) {

  //std::cout << "Cuda add of\n      L " << tensor_x.name << "  &  R " << tensor_w.name << "\n";
    
  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;


  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(Ldims);
  float dims_prod = tensor_x->dims_prod;


  float* device_y = get_from_pool(thread_id, dims_prod, "eq");


  int grid_size, block_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  
  tensor_x->Sync();
  tensor_w->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  equal_forward<<<grid_size, block_size, 0, stream>>>(device_y, device_x, device_w, dims_prod);
  
  
  Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");  
  new_tensor->AttrNodes(tensor_x, tensor_w, equal_op);
  return new_tensor;
}


__global__ void hadamard_kernel(float *y, const float *x,
                            const float *w, int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dims_prod)
        y[i] = x[i] * w[i];
}

extern "C" Tensor *CudaHadamard(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w, int thread_id) {

  //std::cout << "      L " << tensor_x.name << "  &  R " << tensor_w.name << "\n";
    
  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;

  float dims_prod = tensor_x->dims_prod;


  cudaStream_t stream = ThreadsStream[thread_id];
  if (Ldims!=Rdims) //Then broadcast
  { //TODO: change kernel instead
    bool first_iter = true;
    while (Ldims.size()>Rdims.size())
    {
      float tgt_dim_size = Ldims[Rdims.size()];
      float aux_size = DimsProd(Rdims);
      float *aux_tensor, *aux_free;
      cudaMalloc(&aux_tensor, aux_size*tgt_dim_size*sizeof(float));
      cudaMemset(aux_tensor, 0, aux_size*tgt_dim_size*sizeof(float));
      
      int grid_size = dims_prod;
      int block_size = 32;
      size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
      repeat_interleave_kernel_last_dim<<<grid_size, block_size, shared_mem_size, stream>>>(device_w, aux_tensor, aux_size, tgt_dim_size);

      if (!first_iter)
      {
        aux_free = device_w;
        cudaCheck(cudaFree(aux_free));
      }
      device_w = aux_tensor;
      Rdims.push_back(tgt_dim_size);
      first_iter=false;
    }

    while (Ldims.size()<Rdims.size())
    {
      float tgt_dim_size = Rdims[Ldims.size()];
      float aux_size = DimsProd(Ldims);
      float *aux_tensor, *aux_free;
      cudaMalloc(&aux_tensor, aux_size*tgt_dim_size*sizeof(float));
      cudaMemset(aux_tensor, 0, aux_size*tgt_dim_size*sizeof(float));
      
      int grid_size = dims_prod;
      int block_size = 32;
      size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
      repeat_interleave_kernel_last_dim<<<grid_size, block_size, shared_mem_size, stream>>>(device_x, aux_tensor, aux_size, tgt_dim_size);

      if (!first_iter)
      {
        aux_free = device_x;
        cudaCheck(cudaFree(aux_free));
      }
      device_x = aux_tensor;
      
      Ldims.push_back(tgt_dim_size);
      
      dims_prod = DimsProd(Ldims);
      first_iter=false;
    }
  }


  float *device_y = get_from_pool(thread_id, dims_prod, "hadamard");


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  tensor_x->Sync();
  tensor_w->Sync();
  hadamard_kernel<<<grid_size, block_size, 0, stream>>>(device_y, device_x, device_w, dims_prod);
  //PrintTensorF(device_y, 2, 2);



  Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");
  new_tensor->AttrNodes(tensor_x, tensor_w, hadamard_op);
  return new_tensor;
}



extern "C" void *CudaDiv(int is_forward_func,
                          Tensor *tensor_x, Tensor *tensor_w, int thread_id) {
  
  //std::cout << "TENSOR TENSOR DIV" << "\n";
  
  std::vector<float> Ldims, Rdims;
  Ldims = tensor_x->dims;
  Rdims = tensor_w->dims;
  float *device_x = tensor_x->tensor_ptr;
  float *device_w = tensor_w->tensor_ptr;
  float dims_prod, R_dims_prod;
  dims_prod = tensor_x->dims_prod;
  R_dims_prod = tensor_w->dims_prod;


  cudaStream_t stream = ThreadsStream[thread_id];
  if (Ldims!=Rdims) //Then broadcast
  { //TODO: change kernel instead
    bool first_iter = true;
    while (Ldims.size()>Rdims.size())
    {
      float tgt_dim_size = Ldims[Rdims.size()];
      float aux_size = DimsProd(Rdims);
      float *aux_tensor, *aux_free;
      cudaMalloc(&aux_tensor, aux_size*tgt_dim_size*sizeof(float));
      cudaMemset(aux_tensor, 0, aux_size*tgt_dim_size*sizeof(float));
      
      int grid_size = dims_prod;
      int block_size = 32;
      size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
      repeat_interleave_kernel_last_dim<<<grid_size, block_size, shared_mem_size, stream>>>(device_w, aux_tensor, aux_size, tgt_dim_size);

      if (!first_iter)
      {
        aux_free = device_w;
        cudaCheck(cudaFree(aux_free));
      }
      device_w = aux_tensor;
      Rdims.push_back(tgt_dim_size);
      first_iter=false;
    }

    while (Ldims.size()<Rdims.size())
    {
      float tgt_dim_size = Rdims[Ldims.size()];
      float aux_size = DimsProd(Ldims);
      float *aux_tensor, *aux_free;
      cudaMalloc(&aux_tensor, aux_size*tgt_dim_size*sizeof(float));
      cudaMemset(aux_tensor, 0, aux_size*tgt_dim_size*sizeof(float));
      
      int grid_size = dims_prod;
      int block_size = 32;
      size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
      repeat_interleave_kernel_last_dim<<<grid_size, block_size, shared_mem_size, stream>>>(device_x, aux_tensor, aux_size, tgt_dim_size);

      if (!first_iter)
      {
        aux_free = device_x;
        cudaCheck(cudaFree(aux_free));
      }
      device_x = aux_tensor;
      
      Ldims.push_back(tgt_dim_size);
      
      dims_prod = DimsProd(Ldims);
      first_iter=false;
    }
  }


  //if (dims_prod!=R_dims_prod)
  //  LogErrorS("Tensors division has tensors of different dimensions.");


  float* device_y = get_from_pool(thread_id, dims_prod,"div");
  


  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  tensor_div<<<grid_size, block_size, shared_mem_size, stream>>>(device_w, device_x, device_y, dims_prod);

  Tensor *new_tensor = createTensor(device_y, Ldims, dims_prod, false, "");  
  new_tensor->AttrNodes(tensor_x, tensor_w, div_op);
  return new_tensor;
}

__global__ void sum_over_last_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int summed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = summed_dim_size;
    
    if (i < dims_prod) {
        int b = i / (C); // b updates only when v reaches it's maximum value
        

        float *summed_b = summed + b;

        float ix = tensor[i];

        atomicAdd(summed_b, ix);        
    }
}



void broadcast_lastdim_add_backward(float *dx, float *dy, int x_size, int y_size)
{

  int leading_dim = y_size/x_size;


  int grid_size, block_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(y_size);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];


  sum_over_last_dim_kernel<<<grid_size, block_size, 0, main_stream>>>(dy, dx, y_size, leading_dim);
}


extern "C" float network_ema(int thread_id, char *scope, char *_ema_network, char *_network, float factor)
{

  std::string ema_network, network;
  ema_network = NamedObjects[_ema_network];
  network = NamedObjects[_network];

  Tensor *ema_tensor, *net_tensor;
  cudaStream_t stream = ThreadsStream[thread_id];


  //std::cout << "\nNETWORK EMA OF " << ema_network << " and " << network << "\n\n";

  //std::cout << "\n\n\n\n\n\n\n\n\n\nNETWORK EMA OF " << ema_network << " and " << network << "\n";



  std::vector<std::string> ema_params, net_params;
  for (const auto &pair : NamedTensorsT)
  {
    std::string param_name = pair.first;
    if (starts_with(param_name.c_str(), ema_network.c_str()))
      ema_params.push_back(param_name);
    
    if (starts_with(param_name.c_str(), network.c_str()))
      net_params.push_back(param_name);
    
  }


  for (const auto &ema_param : ema_params)
  {
    for (const auto &net_param : net_params)
    {
      if (contains_str(net_param, remove_substring(ema_param, ema_network)))
      {
        //std::cout << "MATCHED PARAMETER: " << ema_param  << " and " << net_param << "\n";

        ema_tensor = NamedTensorsT[ema_param];
        net_tensor = NamedTensorsT[net_param];

        if (ema_tensor->dims_prod!=net_tensor->dims_prod)
        {
          std::string _error = "network_ema failed because " + ema_tensor->name + " and " + net_tensor->name + " parameter sizes do not match.";
          LogErrorS(_error);
        } else {

          int grid_size, block_size;
          std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(ema_tensor->dims_prod);
          grid_size = grid_block_mem_sizes[0];
          block_size = grid_block_mem_sizes[1];

          net_tensor->Sync();
          ema_tensor->Sync();

          
          ema_tensor_kernel<<<grid_size, block_size, 0, stream>>>(ema_tensor->tensor_ptr, net_tensor->tensor_ptr, factor, ema_tensor->dims_prod);

        }
      }
    }
  }

  cudaStreamSynchronize(stream);

  //starts_with

  //std::cout  << "\n\n\n\n\n\n\n\n\n\n";

  return 0;
}




float eps = 1e-8;


// Parallelizes over B, C
__global__ void onehot_kernel(const float *tensor,
                           float *probs,
                           int B, int C) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = threadIdx.x;
    
    if (i < B * C) {
        int b = i / (C);
        int v = i % C;

        float *probs_b = probs + b * C;
        int ix = tensor[b];

        float indicator = (v==ix) ? 1.0f : 0.0f;
        probs_b[v] = indicator;
    }
}


extern "C" void *onehot(int thread_id, Tensor *tensor, float num_classes)
{
  //std::cout << "ONEHOT OF " << tensor.name << "\n";

  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims, new_dims;
  dims = tensor->dims;
  new_dims = tensor->dims;
  new_dims.push_back(num_classes);
  
  int B = DimsProd(dims);
  int C = (int)num_classes;

  
  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];


  tensor->Sync();

  float *probs = get_from_pool(thread_id, B*C, "onehot probs");

  cudaStream_t stream = ThreadsStream[thread_id];
  set_to_zero_kernel<<<grid_size, block_size, 0, stream>>>(probs, B*C);

  onehot_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, probs, B, C);
  //grid_size = ceil_div(B*C, block_size);
  //onehot_kernel<<<grid_size, block_size>>>(tensor, probs, B, C);


  Tensor *new_tensor = createTensor(probs, new_dims, DimsProd(new_dims), false, "");
  new_tensor->AttrLNode(tensor, onehot_op);
  return new_tensor;
}


extern "C" float shape(int thread_id, Tensor tensor)
{
  std::cout << "\nTensor \033[95m" << tensor.name<<"/"<<tensor.scopeless_name << "\033[0m:\n   ";
  PrintDims(tensor.dims);

  return 0;
}


extern "C" float printtt(int thread_id, Tensor tensor)
{
  char* tensorName = new char[tensor.name.size() + 1]; // Allocate memory for the C-style string
  std::strcpy(tensorName, tensor.name.c_str()); // Copy the string

  PrintTensor(thread_id, tensorName);

  delete[] tensorName;
  return 0;
}




extern "C" void *repeat_interleave(int thread_id, Tensor tensor, float repeats, float dim)
{
  //std::cout << "REPEAT_interleave OF " << tensor.name << " with " << repeats << " repeats.\n";

  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims, new_dims;
  dims = tensor.dims;
  if (dim<0)
    dim = dims.size()+dim;
  new_dims = tensor.dims;
  new_dims[dim] = new_dims[dim]*repeats;
  
  int B = DimsProd(dims);
  int C = (int)repeats;

  float *probs;

  probs = get_from_pool(thread_id, B*C, "repeat_interleave");
  cudaMemset(probs, 0, B*C*sizeof(float));
  


  int grid_size = B;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  cudaStream_t stream = ThreadsStream[thread_id];
  if (dim==(dims.size()-1))
    repeat_interleave_kernel_last_dim<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, probs, B, C);
  //grid_size = ceil_div(B*C, block_size);
  //onehot_kernel<<<grid_size, block_size>>>(tensor, probs, B, C);


  Tensor *new_tensor = createTensor(probs, new_dims, DimsProd(new_dims), false, "");
  return new_tensor;
}



__global__ void warped_to_probs_single_dim(float *y, const float *x, int C) {
  
    const int warpsPerBlock = blockDim.x / warpSize;
    int tid = threadIdx.x;

    

    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    // one warp one row
    //int row = blockIdx.x * warpsPerBlock + warpId;
    
    if (laneId >= C)
        return;


    
    float sumval = 0.0f;

#pragma unroll
    for (int i = laneId; i < C; i += warpSize)
        sumval += x[i];
    

    float offsetSumval;

#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        __syncwarp();
        offsetSumval = __shfl_down_sync(0xFFFFFFFF, sumval, offset);
        
        sumval += offsetSumval;
    }


    sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);


#pragma unroll
    for (int i = laneId; i < C; i += warpSize)
        y[i] = x[i] / sumval;
}


__global__ void sample_val_from_probs(float *tensor, float *sampled_value, int n, unsigned long long seed) {
    // Get the thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    // Only one thread needs to sample a value
    if (idx == 0) {
        // Setup random generator
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate a random float in the range [0, 1)
        float rand_value = curand_uniform(&state);

        // Perform sampling
        float cumulative_sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            cumulative_sum += tensor[i];
            if (rand_value < cumulative_sum) {
                sampled_value[0] = tensor[i];  // Sampled value
                return;
            }
        }
    }
}

__global__ void sample_from_probs(float *tensor, float *sampled_value, int n, unsigned long long seed) {
    // Get the thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    

    // Only one thread needs to sample a value
    if (idx == 0) {

        // Setup random generator
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate a random float in the range [0, 1)
        float rand_value = curand_uniform(&state);

        // Perform sampling
        float cumulative_sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            cumulative_sum += tensor[i];
            if (rand_value < cumulative_sum) {
                sampled_value[0] = i;  // Sampled value
                return;
            }
        }
    }
}


extern "C" float priority_sample(int thread_id, Tensor *tensor, float max_idx, float seed)
{
  
  float *probs, *sampled, *probs_cpu;
  float ret;
  probs = get_from_pool(thread_id, max_idx, "priority sample");
  sampled = get_from_pool(thread_id, 1, "priority sample");
  probs_cpu = new float[1];


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(max_idx, 32);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  //TODO: optimize to use shared memory and suprass 32 threads.
  cudaStream_t stream = ThreadsStream[thread_id];
  warped_to_probs_single_dim<<<grid_size, block_size, 0, stream>>>(probs, tensor->tensor_ptr, max_idx);


  //unsigned long long seed = get_int_seed();


  sample_from_probs<<<1, 1, 0, stream>>>(probs, sampled, max_idx, seed);


  cudaStreamSynchronize(stream);
  cudaMemcpy(probs_cpu, sampled, 1*sizeof(float), cudaMemcpyDeviceToHost);
  ret = probs_cpu[0];
  delete[] probs_cpu;



  move_to_pool(thread_id, max_idx, probs, "priority sample");
  move_to_pool(thread_id, 1, sampled, "priority sample");


  return ret;
}

extern "C" float priority_sample_val(int thread_id, Tensor *tensor, float max_idx, float seed)
{  
  float *probs, *sampled, *probs_cpu;
  float ret;
  probs = get_from_pool(thread_id, max_idx, "priority sample");
  sampled = get_from_pool(thread_id, 1, "priority sample");
  probs_cpu = new float[1];


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(max_idx, 32);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  //TODO: optimize to use shared memory and suprass 32 threads.
  cudaStream_t stream = ThreadsStream[thread_id];
  warped_to_probs_single_dim<<<grid_size, block_size, 0, stream>>>(probs, tensor->tensor_ptr, max_idx);


  //unsigned long long seed = get_int_seed();

  grid_block_mem_sizes = CalculateGridAndBlockSizes(max_idx);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  sample_val_from_probs<<<1, 1, 0, stream>>>(probs, sampled, max_idx, seed);


  cudaStreamSynchronize(stream);
  cudaMemcpy(probs_cpu, sampled, 1*sizeof(float), cudaMemcpyDeviceToHost);
  ret = probs_cpu[0];
  delete[] probs_cpu;

  move_to_pool(thread_id, max_idx, probs, "priority sample");
  move_to_pool(thread_id, 1, sampled, "priority sample");


  return ret;
}


__global__ void warped_to_probs_single_dim_pow(float *y, const float *x, float alpha, int C) {
  
    const int warpsPerBlock = blockDim.x / warpSize;
    int tid = threadIdx.x;

    
    
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    // one warp one row
    //int row = blockIdx.x * warpsPerBlock + warpId;
    
    if (laneId >= C)
        return;


    
    float sumval = 0.0f;

#pragma unroll
    for (int i = laneId; i < C; i += warpSize)
        sumval += pow(x[i], alpha);
    

    float offsetSumval;

#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        __syncwarp();
        offsetSumval = __shfl_down_sync(0xFFFFFFFF, sumval, offset);
        
        sumval += offsetSumval;
    }


    sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);


#pragma unroll
    for (int i = laneId; i < C; i += warpSize)
        y[i] = x[i] / sumval;
    
}




extern "C" float importance_sample_idx(int thread_id, Tensor *tensor, float max_idx, float alpha, float beta, float seed)
{  
  
  float *probs, *sampled, *probs_cpu;
  float ret;
  probs = get_from_pool(thread_id, tensor->dims_prod, "priority sample probs");
  sampled = get_from_pool(thread_id, 1, "priority sample sampled");
  probs_cpu = new float[1];


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(max_idx, 32);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  //TODO: optimize to use shared memory and suprass 32 threads.
  cudaStream_t stream = ThreadsStream[thread_id];
  
  warped_to_probs_single_dim_pow<<<grid_size, block_size, 0, stream>>>(probs, tensor->tensor_ptr, alpha, max_idx);


  sample_from_probs<<<1, 1, 0, stream>>>(probs, sampled, max_idx, seed);
  

  cudaStreamSynchronize(stream);
  cudaMemcpy(probs_cpu, sampled, 1*sizeof(float), cudaMemcpyDeviceToHost);
  ret = probs_cpu[0];
  
  delete[] probs_cpu;


  move_to_pool(thread_id, tensor->dims_prod, probs, "priority sample");
  move_to_pool(thread_id, 1, sampled, "priority sample");


  return ret;
}


__global__ void is_w_kernel(float *is_w_ptr, const float *probs, const float *idx, float beta, float max_idx)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lid = tid % warpSize;

  if (lid>=max_idx)
    return;

  float eps = 1e-6;

  float is_w = pow(1/(probs[(int)idx[0]]*max_idx + eps), beta);
  float iter_is_w;



  float max_is_w = -INFINITY;

#pragma unroll
  for (int i=lid; i<max_idx; i+=warpSize)
  {

    iter_is_w = pow(1/(probs[i]*max_idx + eps), beta);
    max_is_w = fmaxf(iter_is_w, max_is_w);
  }

  
  float warp_is_w;
#pragma unroll
  for (int mask=warpSize/2; mask > 0; mask>>=1)
  {
    __syncwarp();
    warp_is_w = __shfl_down_sync(0xFFFFFFFF, max_is_w, mask);
    max_is_w = fmaxf(max_is_w, warp_is_w);
  }
  max_is_w = __shfl_sync(0xFFFFFFFF, max_is_w, 0);


  is_w_ptr[0] = is_w / max_is_w;
}



extern "C" float importance_sample_weight(int thread_id, Tensor *tensor, float max_idx, float alpha, float beta, float seed)
{  
  float *probs, *sampled, *is_w_cpu, *is_w;
  float ret;
  probs = get_from_pool(thread_id, tensor->dims_prod, "importance_sample_weight probs");
  sampled = get_from_pool(thread_id, 1, "importance_sample_weight sampled");
  is_w = get_from_pool(thread_id, 1, "importance_sample_weight is_w");
  is_w_cpu = new float[1];


  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(max_idx, 32);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  //TODO: optimize to use shared memory and suprass 32 threads.
  cudaStream_t stream = ThreadsStream[thread_id];
  warped_to_probs_single_dim_pow<<<grid_size, block_size, 0, stream>>>(probs, tensor->tensor_ptr, alpha, max_idx);


  sample_from_probs<<<1, 1, 0, stream>>>(probs, sampled, max_idx, seed);

  
  is_w_kernel<<<grid_size, block_size, 0, stream>>>(is_w, probs, sampled, beta, max_idx);

  cudaStreamSynchronize(stream);
  cudaMemcpy(is_w_cpu, is_w, 1*sizeof(float), cudaMemcpyDeviceToHost);
  ret = is_w_cpu[0];
  delete[] is_w_cpu;

  

  move_to_pool(thread_id, tensor->dims_prod, probs, "importance_sample_weight");
  move_to_pool(thread_id, 1, sampled, "importance_sample_weight");
  move_to_pool(thread_id, 1, is_w, "importance_sample_weight");


  return ret;
}



__global__ void mean_over_semilast_dim_kernel(const float *x, float *y, const int dims_prod, const int T, const int C, const int warps_per_block)
{
  int tid = threadIdx.x;
  int b = blockIdx.x;

  if (b>=dims_prod)
    return;

  int warpId = tid / warpSize;
  int laneId = tid % warpSize;

  for (int warp_tile=0; warp_tile<ceilf(C/(float)warps_per_block); ++warp_tile)
  {
    int c = warp_tile*warps_per_block + warpId;
    __syncwarp();
    float sumval=0.0f;
    if (c<C)
    {
      for (int lane_tile=laneId; lane_tile<T; lane_tile+=warpSize)
        sumval += x[b*T*C + lane_tile*T + c];

      float mask_sumval;
      for(int mask=warpSize/2; mask>0; mask>>=1)
      {
        __syncwarp();
        mask_sumval = __shfl_down_sync(0xFFFFFFFF, sumval, mask);
        sumval+=mask_sumval;
      }
      sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);

      y[b*C + c] = sumval/T;
    }
  }
}

//TODO: mean over axis
extern "C" void *mean(int thread_id, Tensor *tensor, float first_dim, ...)
{
  //std::cout << "MEAN OF " << tensor->name << "\n";


  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float *summed;

  cudaStream_t stream = ThreadsStream[thread_id];

  va_list args;
  va_start(args, first_dim);

  if (first_dim==TERMINATE_VARARG)
  { 
    va_end(args);
    float *ret;
    int dims_prod = DimsProd(dims);

    summed = new float[dims_prod];
    cudaCheck(cudaMemcpyAsync(summed, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToHost, stream));

    cudaCheck(cudaMalloc(&ret, 1*sizeof(float)));
  
    float tensor_sum=0;
    for(int i=0; i<dims_prod; i++)
      tensor_sum += summed[i];
    tensor_sum = tensor_sum/tensor->dims_prod;
    
    delete[] summed;
  
    float *aux = new float[1];
    aux[0] = tensor_sum;
    cudaCheck(cudaMemcpyAsync(ret, aux, 1*sizeof(float), cudaMemcpyHostToDevice, stream));
    delete[] aux;
  
    std::vector<float> new_dims;
    new_dims.push_back(1.0f);
  
    Tensor *new_tensor = createTensor(ret, new_dims, 1.0f, false, "");
    new_tensor->op=mean_op;
    new_tensor->AttrLNode(tensor, mean_op);
    return new_tensor;
  }


  std::vector<float> sum_dims, new_dims;
  if (first_dim<0)
    first_dim = dims.size()+first_dim;
  sum_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions??? (mean)");
      std::cout << "Input tensor dims:" << "\n";
      PrintDims(tensor->dims);
      std::cout << "Mean dims:" << "\n";
      PrintDims(sum_dims);
      return nullptr;
    }

    float dim = va_arg(args, float);
    
    if (dim==TERMINATE_VARARG)
      break;
    if (in_float_vec(dim, sum_dims))  
    {
      std::string _error = "Dim "+std::to_string(dim) + " duplicated at tensor.mean() operation.";
      LogErrorS(_error);
      return nullptr;
    }
    if (dim<0)
      dim = dims.size()+dim;
    sum_dims.push_back(dim);
  }
  va_end(args);
  
  
  float summed_dim;
  for (int i=0; i<dims.size(); i++)
    if (!in_float_vec(i, sum_dims))
      new_dims.push_back(dims[i]);
    else
      summed_dim=dims[i];


  int dims_prod = DimsProd(dims);
  int new_dims_prod = DimsProd(new_dims);

  
  summed = get_from_pool(thread_id, new_dims_prod, "mean");
  cudaMemset(summed, 0, new_dims_prod * sizeof(float));

  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nSummed dim size: " << summed_dim << "\n\n";

  
  
  

  if (sum_dims[0]==(dims.size()-2))
  {
    std::vector<float> _dims = RemoveLastDim(RemoveLastDim(dims));
    dims_prod = DimsProd(_dims);

    int warps_per_block = THREADS_PER_BLOCK/WARP_SIZE;
    //warps_per_block = fminf(warps_per_block, dims[dims.size()-2]);
    

    mean_over_semilast_dim_kernel<<<dims_prod, warps_per_block*WARP_SIZE, 0, stream>>>(tensor_ptr, summed, dims_prod, dims[dims.size()-2], dims[dims.size()-1], warps_per_block);

    Tensor *new_tensor = createTensor(summed, new_dims, new_dims_prod, false, "");
    new_tensor->AttrLNode(tensor, mean_over_semilast_dim_op);
    new_tensor->scalar = dims[dims.size()-2];
    return new_tensor;
  }

  /*
  if (dims.size()==1)
  {
    sum_single_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod);
    new_dims = {1.0f};
  }
  else if (sum_dims[0]==(dims.size()-1))
    sum_over_last_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod, summed_dim);
  if (sum_dims[0]==(dims.size()-2))
    sum_over_semilast_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor_ptr, summed, dims_prod, dims[dims.size()-1], dims[dims.size()-2]);


  Tensor *new_tensor = createTensor(summed, new_dims, DimsProd(new_dims), false, "");
  return new_tensor;
  */
  LogErrorS("Mean of specific dim is not implemented yet.");
  return nullptr;
}



__global__ void mean_over_semilast_dim_backward_kernel(float *dx, const float *dy, const int dims_prod, const int T, const int C)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if(idx>=dims_prod)
    return;



  int b = idx / (T*C);
  int t = (idx/C) % T;
  int c = idx % C;


  dx[b*T*C + t*C + c] = dy[b*C + c];
}

void mean_over_semilast_dim_backward(float *dx, float *dy, Tensor *node)
{
  std::vector<float> dims = node->L_Node->dims;
  float x_dims_prod = node->L_Node->dims_prod;
  float y_dims_prod = node->dims_prod;


  mean_over_semilast_dim_backward_kernel<<<std::ceil(x_dims_prod/(float)THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, main_stream>>>(dx, dy,  x_dims_prod, dims[dims.size()-2], dims[dims.size()-1]);
}



__global__ void sum_single_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  int C = dims_prod;
  
  if (i < dims_prod) {
    int b = i / (C); // b updates only when v reaches it's maximum value
    int v = i % C;


    float ix = tensor[i];

    atomicAdd(summed, ix);
  }
}


__global__ void sum_over_semilast_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int last_dim_size, int summed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = last_dim_size;
    int D = summed_dim_size*last_dim_size;
    
    if (i < dims_prod) {
        int b = i / C; // b updates only when v reaches it's maximum value
        int d = i / D;
        int v = i % C;
        // i = b*C + v

        float *summed_b = summed + v + d*C;

        float ix = tensor[i];

        atomicAdd(summed_b, ix);        
    }
}




extern "C" void *sum(int thread_id, Tensor tensor, float first_dim, ...)
{
  //std::cout << "SUM OF " << tensor.name << "\n";


  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  float *summed;

  cudaStream_t stream = ThreadsStream[thread_id];

  va_list args;
  va_start(args, first_dim);

  if (first_dim==TERMINATE_VARARG)
  {
    va_end(args);
    float *ret;
    int dims_prod = DimsProd(dims);

    summed = new float[dims_prod];
    cudaCheck(cudaMemcpyAsync(summed, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToHost, stream));

    cudaCheck(cudaMalloc(&ret, 1*sizeof(float)));
  
    float tensor_sum=0;
    for(int i=0; i<dims_prod; i++)
      tensor_sum += summed[i];
    
    delete[] summed;
  
    float *aux = new float[1];
    aux[0] = tensor_sum;
    cudaCheck(cudaMemcpy(ret, aux, 1*sizeof(float), cudaMemcpyHostToDevice));  
    delete[] aux;
  
    std::vector<float> new_dims;
    new_dims.push_back(1.0f);
  
    Tensor *new_tensor = createTensor(ret, new_dims, 1.0f, false, "");
    new_tensor->op=sum_op;
    return new_tensor;
  }


  std::vector<float> sum_dims, new_dims;
  if (first_dim<0)
    first_dim = dims.size()+first_dim;
  sum_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions??? (sum)");
      return nullptr;
    }

    float dim = va_arg(args, float);
    
    if (dim==TERMINATE_VARARG)
      break;
    if (in_float_vec(dim, sum_dims))  
    {
      std::string _error = "Dim "+std::to_string(dim) + " duplicated at tensor.sum() operation.";
      LogErrorS(_error);
      return nullptr;
    }
    if (dim<0)
      dim = dims.size()+dim;
    sum_dims.push_back(dim);
  }
  va_end(args);
  
  
  float summed_dim;
  for (int i=0; i<dims.size(); i++)
    if (!in_float_vec(i, sum_dims))
      new_dims.push_back(dims[i]);
    else
      summed_dim=dims[i];


  int dims_prod = DimsProd(dims);
  int new_dims_prod = DimsProd(new_dims);

  
  summed = get_from_pool(thread_id, new_dims_prod, "summed");
  cudaMemset(summed, 0, new_dims_prod * sizeof(float));

  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nSummed dim size: " << summed_dim << "\n\n";

  
  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  
  if (dims.size()==1)
  {
    sum_single_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, summed, dims_prod);
    new_dims = {1.0f};
  }
  else if (sum_dims[0]==(dims.size()-1))
    sum_over_last_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, summed, dims_prod, summed_dim);
  if (sum_dims[0]==(dims.size()-2))
    sum_over_semilast_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, summed, dims_prod, dims[dims.size()-1], dims[dims.size()-2]);


  Tensor *new_tensor = createTensor(summed, new_dims, DimsProd(new_dims), false, "");
  new_tensor->op=sum_op;
  return new_tensor;
}


__device__ float atomicMul(float* address, float val) {
    int *addr_as_int = (int *)address;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(val * __int_as_float(assumed)));
    } while (assumed != old);
    return __int_as_float(old);
}


__global__ void prod_single_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = dims_prod;
    
    if (i < dims_prod) {
        int b = i / (C); // b updates only when v reaches it's maximum value
        int v = i % C;
        // i = b*C + v


        float ix = tensor[i];

        atomicMul(summed, ix);        
    }
}

__global__ void prod_over_last_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int summed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = summed_dim_size;
    
    if (i < dims_prod) {
        int b = i / (C); // b updates only when v reaches it's maximum value
        int v = i % C;
        // i = b*C + v

        float *summed_b = summed + b;

        float ix = tensor[i];

        atomicMul(summed_b, ix);        
    }
}

__global__ void prod_over_semilast_dim_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int last_dim_size, int summed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = last_dim_size;
    int D = summed_dim_size*last_dim_size;
    
    if (i < dims_prod) {
        int b = i / C; // b updates only when v reaches it's maximum value
        int d = i / D;
        int v = i % C;
        // i = b*C + v

        float *summed_b = summed + v + d*C;

        float ix = tensor[i];

        atomicMul(summed_b, ix);        
    }
}

extern "C" void *prod(int thread_id, Tensor tensor, float first_dim, ...)
{
  //std::cout << "PROD OF " << tensor.name << "\n";


  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  float *summed;

  cudaStream_t stream = ThreadsStream[thread_id];

  va_list args;
  va_start(args, first_dim);

  if (first_dim==TERMINATE_VARARG)
  {
    va_end(args);
    int dims_prod = DimsProd(dims);

    summed = get_from_pool(thread_id, dims_prod, "prod all dims");
    cudaMemcpyAsync(summed, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    float tensor_sum=0;
    for(int i=0; i<dims_prod; i++)
      tensor_sum += summed[i];
    tensor_sum = tensor_sum;

    std::cout << "prod: " << tensor_sum << "\n";

    return summed;
  }


  std::vector<float> sum_dims, new_dims;
  if (first_dim<0)
    first_dim = dims.size()+first_dim;
  sum_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions??? (prod)");
      return nullptr;
    }

    float dim = va_arg(args, float);
    
    if (dim==TERMINATE_VARARG)
      break;
    if (in_float_vec(dim, sum_dims))  
    {
      std::string _error = "Dim "+std::to_string(dim) + " duplicated at tensor.sum() operation.";
      LogErrorS(_error);
      return nullptr;
    }
    if (dim<0)
      dim = dims.size()+dim;
    sum_dims.push_back(dim);
  }
  va_end(args);
  
  
  float summed_dim;
  for (int i=0; i<dims.size(); i++)
    if (!in_float_vec(i, sum_dims))
      new_dims.push_back(dims[i]);
    else
      summed_dim=dims[i];


  int dims_prod = DimsProd(dims);
  int new_dims_prod = DimsProd(new_dims);

  
  float *init_prod = new float[new_dims_prod];
  init_prod = make_ones_float(new_dims_prod);
  
  summed = get_from_pool(thread_id, new_dims_prod, "prod");
  cudaMemcpyAsync(summed, init_prod, new_dims_prod * sizeof(float), cudaMemcpyHostToDevice, stream);
  delete[] init_prod;

  //PrintTensorF(summed, new_dims_prod,1);

  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nSummed dim size: " << summed_dim << "\n\n";

  
  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  
  if (dims.size()==1)
  {
    prod_single_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, summed, dims_prod);
    new_dims = {1.0f};
  }
  else if (sum_dims[0]==(dims.size()-1))
  {
    prod_over_last_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, summed, dims_prod, summed_dim);
    //std::cout << "prod_over_last_dim_kernel" << "\n";
  }
  if (sum_dims[0]==(dims.size()-2))
    prod_over_semilast_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, summed, dims_prod, dims[dims.size()-1], dims[dims.size()-2]);


  Tensor *new_tensor = createTensor(summed, new_dims, DimsProd(new_dims), false, "");
  return new_tensor;
}


__global__ void max_over_last_dim_kernel(const float *tensor,
                           float *maxed,
                           int dims_prod, int maxed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = maxed_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *max_b = maxed + b;

        float ix = tensor[i];


        unsigned int *const addr_as_ui = (unsigned int *)max_b;
        unsigned int old = *addr_as_ui, assumed;
        do {
          assumed = old;
          if (__uint_as_float(assumed) >= ix) break;
          old = atomicCAS(addr_as_ui, assumed, __float_as_uint(ix));
        } while (assumed != old);
    }
}

__global__ void max_over_semilast_dim_kernel(const float *tensor,
                           float *maxed,
                           int dims_prod, int last_dim_size, int maxed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = last_dim_size;
    int D = maxed_dim_size*last_dim_size;
    
    
    if (i < dims_prod) {
        int b = i / C;
        int d = i / D;
        int v = i % C;
        // i = b * C + v

        float *max_b = maxed + v + d*C;

        float ix = tensor[i];

        unsigned int *const addr_as_ui = (unsigned int *)max_b;
        unsigned int old = *addr_as_ui, assumed;
        do {
          assumed = old;
          if (__uint_as_float(assumed) >= ix) break;
          old = atomicCAS(addr_as_ui, assumed, __float_as_uint(ix));
        } while (assumed != old);
    }
}


extern "C" void *tmax(int thread_id, Tensor *tensor, float first_dim, ...) 
{ //TODO: automatic type detection for max and min (float vs tensor)
  
  //std::cout << "MAX OF " << tensor.name << "\n";
  

  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float *summed;

  cudaStream_t stream = ThreadsStream[thread_id];
  tensor->Sync();

  va_list args;
  va_start(args, first_dim);

  if (first_dim==TERMINATE_VARARG)
  {
    va_end(args);
    int dims_prod = DimsProd(dims);

    summed = get_from_pool(thread_id, dims_prod, "tmax all dims");
    cudaMemcpyAsync(summed, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    float tensor_sum=0;
    for(int i=0; i<dims_prod; i++)
      tensor_sum += summed[i];
    tensor_sum = tensor_sum;

    std::cout << "Sum: " << tensor_sum << "\n";

    return summed;
  }


  std::vector<float> sum_dims, new_dims;
  if (first_dim<0)
    first_dim = dims.size()+first_dim;
  sum_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions??? (max)");
      return nullptr;
    }

    float dim = va_arg(args, float);
    
    if (dim==TERMINATE_VARARG)
      break;
    if (in_float_vec(dim, sum_dims))  
    {
      std::string _error = "Dim "+std::to_string(dim) + " duplicated at tensor.sum() operation.";
      LogErrorS(_error);
      return nullptr;
    }
    if (dim<0)
      dim = dims.size()+dim;
    sum_dims.push_back(dim);
  }
  va_end(args);
  
  
  float summed_dim;
  for (int i=0; i<dims.size(); i++)
    if (!in_float_vec(i, sum_dims))
      new_dims.push_back(dims[i]);
    else
      summed_dim=dims[i];


  int dims_prod = DimsProd(dims);
  int new_dims_prod = DimsProd(new_dims);

  
  summed = get_from_pool(thread_id, new_dims_prod, "tmax");
  cudaMemset(summed, 0, new_dims_prod * sizeof(float));


  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nSummed dim size: " << summed_dim << "\n\n";

  
  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  // AtomicMax does not handle negative numbers, so gambiarra. :D (1 hour for this)
  vec_add<<<grid_size, block_size, shared_mem_size, stream>>>(50000, tensor_ptr, tensor_ptr, dims_prod);
  if (sum_dims[0]==(dims.size()-1))
    max_over_last_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, summed, dims_prod, summed_dim);
  if (sum_dims[0]==(dims.size()-2))
    max_over_semilast_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, summed, dims_prod, dims[dims.size()-1], dims[dims.size()-2]);
  vec_sub<<<grid_size, block_size, shared_mem_size, stream>>>(50000, summed, summed, new_dims_prod);
  vec_sub<<<grid_size, block_size, shared_mem_size, stream>>>(50000, tensor_ptr, tensor_ptr, dims_prod);


  Tensor *new_tensor = createTensor(summed, new_dims, DimsProd(new_dims), false, "");
  new_tensor->AttrLNode(tensor, max_op);
  return new_tensor;
}


__global__ void argmax_over_last_dim_kernel(const float *tensor,
                           float *maxed, float *argmaxed,
                           int dims_prod, int maxed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = maxed_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *max_b = maxed + b;
        float *argmax_b = argmaxed + b;

        float ix = tensor[i];

        // max
        int *addr_as_int = (int *)max_b;
        int old_int = *addr_as_int, assumed_int;
        float old_val;
        do {
            assumed_int = old_int;
            old_val = __int_as_float(assumed_int);
            if (old_val >= ix) break;
            old_int = atomicCAS(addr_as_int, assumed_int, __float_as_int(ix));
        } while (assumed_int != old_int);

        // argmax
        if (__int_as_float(old_int) < ix) {
            int *addr_as_int_argmax = (int *)argmax_b;
            atomicExch(addr_as_int_argmax, __float_as_int((float)v));
        }
      }
}

extern "C" void *argmax(int thread_id, Tensor *tensor, float first_dim, ...) 
{
  //std::cout << "ARGMAX OF " << tensor->name << " at thread: " << thread_id << "\n";
  cudaCheck(cudaGetLastError());

  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float *maxed, *argmaxed;


  va_list args;
  va_start(args, first_dim);

  if (first_dim==TERMINATE_VARARG)
  {
    va_end(args);
    LogErrorS("Argmax is only supported at dim -1.");
    return nullptr;
  }


  std::vector<float> sum_dims, new_dims;
  if (first_dim<0)
    first_dim = dims.size()+first_dim;
  sum_dims.push_back(first_dim);

  for (int i=0; i<10; i++)
  {
    if (i==9)
    {
      LogErrorS("A tensor with 10 dimensions??? (argmax)");
      return nullptr;
    }

    float dim = va_arg(args, float);
    
    if (dim==TERMINATE_VARARG)
      break;
    if (in_float_vec(dim, sum_dims))  
    {
      std::string _error = "Dim "+std::to_string(dim) + " duplicated at tensor.argmax() operation.";
      LogErrorS(_error);
      return nullptr;
    }
    if (dim<0)
      dim = dims.size()+dim;
    sum_dims.push_back(dim);
  }
  va_end(args);
  
  
  float maxed_dim;
  for (int i=0; i<dims.size(); i++)
    if (!in_float_vec(i, sum_dims))
      new_dims.push_back(dims[i]);
    else
      maxed_dim=dims[i];


  int dims_prod = DimsProd(dims);
  int new_dims_prod = DimsProd(new_dims);

    
  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(new_dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  

  tensor->Sync();
  maxed = get_from_pool(thread_id, new_dims_prod, "argmax maxed");
  argmaxed = get_from_pool(thread_id, new_dims_prod, "argmax");

  cudaStream_t stream = ThreadsStream[thread_id];
  set_to_zero_kernel<<<grid_size, block_size, 0, stream>>>(maxed, new_dims_prod);
  set_to_zero_kernel<<<grid_size, block_size, 0, stream>>>(argmaxed, new_dims_prod);


  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nMaxed dim size: " << summed_dim << "\n\n";

  
  grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];

  
  vec_add<<<grid_size, block_size, shared_mem_size, stream>>>(50000, tensor_ptr, tensor_ptr, dims_prod);
  if (sum_dims[0]==(dims.size()-1))
    argmax_over_last_dim_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, maxed, argmaxed, dims_prod, maxed_dim);
  //if (sum_dims[0]==(dims.size()-2))
  //  max_over_semilast_dim_kernel<<<grid_size, block_size, shared_mem_size>>>(tensor, maxed, dims_prod, dims[dims.size()-1], dims[dims.size()-2]);
  vec_sub<<<grid_size, block_size, shared_mem_size, stream>>>(50000, tensor_ptr, tensor_ptr, dims_prod);

  if(thread_id==0)
  {
    //std::cout << "maxed is " << new_dims_prod << "\n";
    move_to_pool(thread_id, new_dims_prod, maxed, "argmax maxed");
  }
  else
    cudaFree(maxed);
  

  Tensor *new_tensor = createTensor(argmaxed, new_dims, DimsProd(new_dims), false, "");
  new_tensor->AttrLNode(tensor, argmax_op);
  cudaCheck(cudaGetLastError());
  return new_tensor;
}

__global__ void topk_kernel(const float *tensor, float *topk,
                           float *maxed, float *argmaxed,
                           int dims_prod, int maxed_dim_size,
                           int j, int k) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = maxed_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *max_b = maxed + b;
        float *argmax_b = argmaxed + b;
        float *topk_b = topk + b*k + j;

        float ix = tensor[i];

        // max
        int *addr_as_int = (int *)max_b;
        int old_int = *addr_as_int, assumed_int;
        float old_val;
        do {
            assumed_int = old_int;
            old_val = __int_as_float(assumed_int);
            if (old_val >= ix) break;
            old_int = atomicCAS(addr_as_int, assumed_int, __float_as_int(ix));
        } while (assumed_int != old_int);

        // argmax & topk
        if (__int_as_float(old_int) < ix) {
            int *addr_as_int_argmax = (int *)argmax_b;
            atomicExch(addr_as_int_argmax, __float_as_int((float)v));

            int *addr_as_int_topk = (int *)topk_b;
            atomicExch(addr_as_int_topk, __float_as_int((float)v));
        }
      }
}

__global__ void topk_erase_argmax_aux_kernel(float *tensor,
                           float *argmaxed, int dims_prod, int maxed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = maxed_dim_size;
    
    if (i < dims_prod) {
        int b = i / C;
        int v = i % C;
        // i = b * C + v

        float *tensor_b = tensor + b * C;

        float ix = argmaxed[b];

        float indicator = (v==ix) ? 0 : ix;
        if (v==ix)
          tensor_b[v] = 0;
      }
}

extern "C" void *topk(int thread_id, Tensor tensor, float k) 
{
  std::cout << "TOPK OF " << tensor.name << "\n";

  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  float *maxed, *argmaxed, *topk, *tensor_copy;


  std::vector<float> new_dims = RemoveLastDim(dims);
  std::vector<float> topk_dims = RemoveLastDim(dims);
  float new_dims_prod = DimsProd(new_dims);
  int dims_prod = DimsProd(dims);
  topk_dims.push_back(k);
  float topk_dims_prod = DimsProd(topk_dims);

  float maxed_dim = dims[dims.size()-1];

  cudaStream_t stream = ThreadsStream[thread_id];
  
  cudaMalloc(&maxed, new_dims_prod*sizeof(float));
  cudaMalloc(&argmaxed, new_dims_prod*sizeof(float));
  cudaMalloc(&topk, topk_dims_prod * sizeof(float));
  cudaMalloc(&tensor_copy, dims_prod*sizeof(float));
  cudaMemset(maxed, 0, new_dims_prod*sizeof(float));
  cudaMemset(argmaxed, 0, new_dims_prod*sizeof(float));
  cudaMemset(topk, 0, topk_dims_prod * sizeof(float));
  cudaMemcpyAsync(tensor_copy, tensor_ptr, dims_prod*sizeof(float), cudaMemcpyDeviceToDevice, stream);


  //std::cout << "\n\nDims prod: " << dims_prod << "\nNew dims prod: " << new_dims_prod << "\nMaxed dim size: " << summed_dim << "\n\n";

  
  int grid_size = dims_prod;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);

  
  vec_add<<<grid_size, block_size, shared_mem_size, stream>>>(50000, tensor_copy, tensor_copy, dims_prod);
  
  for (int i=0; i<k; i++)
  {
    //std::cout << "Top k at iter:" << i << "\n";
    topk_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_copy, topk, maxed, argmaxed, dims_prod, maxed_dim, i, k);
    //PrintTensorF(maxed, 3, 1);
    //PrintTensorF(argmaxed, 3, 1);
    //std::cout << "Topk" << "\n";
    //PrintTensorF(topk, 3, k);
    topk_erase_argmax_aux_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_copy, argmaxed, dims_prod, maxed_dim);
    cudaMemset(maxed, 0, new_dims_prod*sizeof(float));
    //std::cout << "\n\n\n\n";
  }
  cudaCheck(cudaFree(tensor_copy));
  cudaCheck(cudaFree(maxed));
  cudaCheck(cudaFree(argmaxed));

  Tensor *new_tensor = createTensor(topk, topk_dims, topk_dims_prod, false, "");
  new_tensor->op=topk_op;
  return new_tensor;
}





extern "C" void *clip(int thread_id, Tensor tensor, float _min, float _max)
{
  float *tensor_ptr = tensor.tensor_ptr;
  std::vector<float> dims = tensor.dims;
  
  int B = DimsProd(dims);

  float* device_y = get_from_pool(thread_id, B,"clip");


  int grid_size = B;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  cudaStream_t stream = ThreadsStream[thread_id];
  tensor_clip<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, device_y, _min, _max, B);

  
  Tensor *new_tensor = createTensor(device_y, dims, tensor.dims_prod, false, "");
  new_tensor->op=clip_op; //TODO: what is the grad of clip?
  return new_tensor;
}


__global__ void gelu_forward_kernel1(const float* inp, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}
__global__ void gelu_backward1(float* dinp, const float* inp, const float* dout, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = (float)inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (float)(local_grad * (float)dout[i]);
    }
}
void gelu_backward(const float* inp, float dims_prod, float* dinp, const float* dout) {

  
  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];

  gelu_backward1<<<grid_size, block_size, 0, main_stream>>>(dinp, inp, dout, dims_prod);
  
}

extern "C" void *gelu(int thread_id, Tensor *tensor)
{
  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;

  std::cout << "GELU AT THREAD " << thread_id << "\n";
  

  float dims_prod = DimsProd(dims);

  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];

  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(dims);
  
  float *y = get_from_pool(thread_id, dims_prod,"gelu");

  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  gelu_forward_kernel1<<<grid_size, block_size, 0, stream>>>(tensor_ptr, y, dims_prod);
  

  
  int is_forward_func=1;
  

  Tensor *new_tensor = createTensor(y, dims, DimsProd(dims), false, "");
  new_tensor->AttrLNode(tensor, gelu_op);
  return new_tensor;
}



__global__ void sigmoid_forward_kernel(const float* inp, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = inp[i];
        out[i] = 1/(1+exp(-x));
    }
}
__global__ void sigmoid_backward_kernel(float* dinp, const float* out, const float* dout, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = (float)out[i];
        float local_grad = x * (1 - x);
        dinp[i] = (float)(local_grad * (float)dout[i]);
    }
}

void sigmoid_backward(const float* out, float dims_prod, float* dinp, const float* dout) {
  
  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];

  sigmoid_backward_kernel<<<grid_size, block_size, 0, main_stream>>>(dinp, out, dout, dims_prod);
  
}

extern "C" void *sigmoid(int thread_id, Tensor *tensor)
{
  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  

  float dims_prod = DimsProd(dims);

  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];

  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(dims);
  
  float *y = get_from_pool(thread_id, dims_prod, "sigmoid");  
  
  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  sigmoid_forward_kernel<<<grid_size, block_size, 0, stream>>>(tensor_ptr, y, dims_prod);
  

  
  int is_forward_func=1;


  Tensor *new_tensor = createTensor(y, dims, DimsProd(dims), false, "");
  new_tensor->AttrLNode(tensor, sigmoid_op);
  return new_tensor;
}



__global__ void sigmoid_add2weights_kernel(const float *xl, const float *wl, const float *xr,
                                           const float *wr, float *out, int B, int C, int OC) {
    int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x_block = blockIdx.x;
  int y_block = blockIdx.y;

  const int _aux = 768;
  

  const int tile_size = 16;

  int row = y_block*tile_size + ty;
  int col = x_block*tile_size + tx;



  float y = 0.0f;


  extern __shared__ float smem[];

  int wl_offset = tile_size*tile_size;
  int xr_offset = wl_offset*2;
  int wr_offset = xr_offset*2;

  
  
  for (int i=0; i < ceilf(C/(float)tile_size); ++i)
  {
    // each tile has a subset of columns to work with
    // tile_tid tells which exact column to use from the subset
    // assume w is transposed already

    int _col  = i * tile_size + tx;
    int _col2 = i * tile_size + ty;
    
    if(row<B && _col<C)
      smem[tx* tile_size +ty] = xl[row*C + _col];
    else
      smem[tx* tile_size +ty] = 0;
    
    if (col<OC && _col2<C)
      smem[wl_offset+ty* tile_size +tx] = wl[col*C + _col2];
    else
      smem[wl_offset+ty* tile_size +tx] = 0;
    
    __syncthreads();


    for(int j=0; j<tile_size; ++j)
      y += smem[j* tile_size +ty] * smem[wl_offset+j* tile_size +tx];
    
    __syncthreads();
    
  }

  if(row<B && col<OC)
    out[row*OC+col] = y;
}



extern "C" void *sigmoid_add2weights(int thread_id, Tensor *tensor_xl, Tensor *tensor_wl, Tensor *tensor_xr, Tensor *tensor_wr)
{
  std::vector<float> Ldims, Rdims;
  Ldims = tensor_xl->dims;
  Rdims = tensor_wl->dims;

  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(Ldims);
  std::vector<float> new_dims = NewDimsOnMult(Ldims, Rdims);
  int input_dims_prod = DimsProd(linear_layer_dims);
  int new_dims_prod = DimsProd(new_dims);

  int B = linear_layer_dims[0];
  int C = new_dims[1];
  int OC = new_dims[1];

  float *device_xl = tensor_xl->tensor_ptr;
  float *device_wl = tensor_wl->tensor_ptr;
  float *device_xr = tensor_xr->tensor_ptr;
  float *device_wr = tensor_wr->tensor_ptr;


  /*
  std::cout << "\nxl" << tensor_xl->name << "\n";
  std::cout << "wl" << tensor_wl->name << "\n";
  std::cout << "xr" << tensor_xr->name << "\n";
  std::cout << "wr" << tensor_wr->name << "\n";
  */

  float *out = get_from_pool(thread_id, new_dims_prod, "sigmoid_add2weights");

  int tile_size = 16;
  dim3 grid_size(std::ceil(B/(float)tile_size), std::ceil(OC/(float)tile_size));
  dim3 block_size(tile_size, tile_size);

  int shared_mem_size = std::min(4 * tile_size*tile_size * sizeof(float), deviceProp.sharedMemPerBlock);;
  

  //std::cout << "PRE SIGMOID KERNEL" << "\n";
  sigmoid_add2weights_kernel<<<grid_size, block_size, shared_mem_size, main_stream>>>
          (
            device_xl, device_wl, device_xr, device_wr,
            out,
            B, C, OC
          );
  //std::cout << "POST SIGMOID KERNEL" << "\n";


  // Add tensor to tree only for the later clean up
  Tensor *l = createTensor(nullptr, new_dims, new_dims_prod, false, "");
  Tensor *r = createTensor(nullptr, new_dims, new_dims_prod, false, "");

  l->AttrNodes(tensor_xl, tensor_wl, add_op);
  r->AttrNodes(tensor_xr, tensor_wr, add_op);

  Tensor *new_tensor = createTensor(out, new_dims, new_dims_prod, false, "");
  new_tensor->AttrNodes(l, r, sigmoid_add2weights_op);
  return new_tensor;
}



void sigmoid_add2weights_backward(Tensor *root, float *dy)
{
  //std::cout << "sigmoid_add2weights_backward" << "\n";
  float *out = root->tensor_ptr;


  Tensor *l, *r, *xl, *wl, *xr, *wr;
  /**/
  l = root->L_Node;
  r = root->R_Node;

  xl = l->L_Node;
  wl = l->R_Node;

  xr = r->L_Node;
  wr = r->R_Node;

  
  float *aux1, *aux2, *aux3, *aux4;
  
  
  cudaMemset(aux1, 0, xl->dims_prod*sizeof(float));
  //cudaMemset(aux2, 0, wl->dims_prod*sizeof(float));
  cudaMemset(aux3, 0, xr->dims_prod*sizeof(float));
  //cudaMemset(aux4, 0, wr->dims_prod*sizeof(float));



  std::string tensor_name = xl->scopeless_name;
  if(var_to_grad.count(tensor_name)>0)
  {
    float *acc_y = var_to_grad[tensor_name];

    int grid_size, block_size;
    std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(xl->dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];
    
    add_inplace<<<grid_size, block_size>>>(acc_y, aux1, xl->dims_prod);

    to_pool(xl->dims_prod, acc_y, "sigmoid_add2weights_backward xl grad");

  } else
    var_to_grad[tensor_name] = aux1;
  

  tensor_name = xr->scopeless_name;
  if(var_to_grad.count(tensor_name)>0)
  {
    float *acc_y = var_to_grad[tensor_name];

    int grid_size, block_size;
    std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(xr->dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];
    
    add_inplace<<<grid_size, block_size>>>(acc_y, aux3, xr->dims_prod);

    to_pool(xl->dims_prod, acc_y, "sigmoid_add2weights_backward xl grad");

  } else
    var_to_grad[tensor_name] = aux3;


  // Free only intermediate pointers, there is no need to free node tensor pointers.


  // No need to free root, weight and leaf nodes.
  to_free_tensor(l);
  to_free_tensor(r);

  
  std::cout << "\n\n\n";
}



__global__ void tanh_forward_kernel(const float* inp, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = tanhf(inp[i]);
}
__global__ void tanh_backward_kernel(float* dinp, const float* out, const float* dout, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = (float)out[i];
        float local_grad = 1 - x*x;
        dinp[i] = (float)(local_grad * (float)dout[i]);
    }
}

void tanh_backward(const float* out, float dims_prod, float* dinp, const float* dout) {
  
  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  

  tanh_backward_kernel<<<grid_size, block_size, 0, main_stream>>>(dinp, out, dout, dims_prod);
  
}

extern "C" void *_tanh(int thread_id, Tensor *tensor)
{
  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  

  float dims_prod = DimsProd(dims);

  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = grid_block_mem_sizes[2];

  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(dims);
  
  float *y = get_from_pool(thread_id, dims_prod, "tanh");

  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  tanh_forward_kernel<<<grid_size, block_size, 0, stream>>>(tensor_ptr, y, dims_prod);
    
  int is_forward_func=1;

  //std::cout << "tanh tensor attribution from " << tensor->name<<"/"<<tensor->scopeless_name << "\n";

  Tensor *new_tensor = createTensor(y, dims, DimsProd(dims), false, "");
  new_tensor->AttrLNode(tensor, tanh_op);
  return new_tensor;
}

__global__ void relu_forward(float* Z, float* A,
                             const float dims_prod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < dims_prod) {
        A[idx] = fmaxf(Z[idx], 0);
    }
}

extern "C" void *relu(int thread_id, Tensor *tensor)
{
  //std::cout << "RELU THREAD IS: " << thread_id << "\n";
  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  std::vector<float> linear_layer_dims = format_LinearLayer_Dims(dims);
  float dims_prod = tensor->dims_prod;

  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  

  float *y = get_from_pool(thread_id, dims_prod, "relu");

  tensor->Sync();
  cudaStream_t stream = ThreadsStream[thread_id];
  relu_forward<<<grid_size, block_size, 0, stream>>>(tensor_ptr, y, dims_prod);



  Tensor *new_tensor = createTensor(y, dims, DimsProd(dims), false, "");
  new_tensor->AttrLNode(tensor, relu_op);
  return new_tensor;
}


__global__ void relu_backward1(float* Z, float* dZ, float* dA,
                                       float N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    
    if (index < N) {
        if (Z[index] > 0) {
            dZ[index] = dA[index];
        }
        else {
            dZ[index] = 0;
        }
    }
}

void relu_backward(float* inp, float dims_prod, float* dinp, float* dout) {

  
  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  relu_backward1<<<grid_size, block_size, 0, main_stream>>>(inp, dinp, dout, dims_prod);
  
}

// warp-level reduction for finding the maximum value
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp-level reduction for summing values
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}



__global__ void softmax_forward_kernel4(const float* inp, float* out, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads

    // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
    // shared memory is used for inter-warp reduction
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block //starred
    int laneId = threadIdx.x % 32; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] e (C,)
    const float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
  #pragma unroll
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
  #pragma unroll
    for (int i = tid; i < C; i += blockDim.x)
        out[idx * C + i] = expf(x[i] - offset);

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    // out[idx, :]
    x = out + idx * C;
    float sumval = 0.0f;
  #pragma unroll
    for (int i = tid; i < C; i += blockDim.x)
        sumval += x[i];
    
    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);

    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
      #pragma unroll
        for (int i = 1; i < warpsPerBlock; ++i)
            val += sumvals[i];
        
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
  #pragma unroll
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;
    }
}



__global__ void online_softmax(const float* inp, float* out, int N, int C) {
    // online softmax paper: http://arxiv.org/abs/1805.02867
    // online softmax reduces loops from 3 to 2
    // which is done by calculating sumval and maxval in one loop
    const int warpsPerBlock = blockDim.x / warpSize;
    int tid = threadIdx.x;

    

    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    // one warp one row
    int row = blockIdx.x * warpsPerBlock + warpId;
    
    if (laneId >= C)
        return;

    if (row >= N)
        return;

    const float* x = inp + row * C;
    float* const y = out + row * C;

    // merge calculating maxval and sumval in one loop
    // which is an arithmetic improvment from online softmax over normal softmax
    float maxval = -INFINITY, sumval = 0.0f, bigger;

#pragma unroll
    for (int i = laneId; i < C; i += warpSize) {
        // when updating the maxval, dynamically updates the previous sumval by
        // multiplying e^{previous_maxval - current_maxval}
        bigger = fmaxf(maxval, x[i]);
        sumval = sumval * expf(maxval - bigger) + expf(x[i] - bigger);
        maxval = bigger;
    }

    // use warp functions instead of cooperative groups for better readibility
    // calculate the warp wised maxval and sumval
    float offsetMaxval, offsetSumval;

#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        __syncwarp();
        offsetMaxval = __shfl_down_sync(0xFFFFFFFF, maxval, offset);
        offsetSumval = __shfl_down_sync(0xFFFFFFFF, sumval, offset);
        if (offsetMaxval > maxval) {
            sumval *= expf(maxval - offsetMaxval);
            maxval = offsetMaxval;
        } else {
            offsetSumval *= expf(offsetMaxval - maxval);
        }
        sumval += offsetSumval;
    }

    // sync the warp wised maxval and sumval
    // which are also the maxval and sumval of one row in C
    maxval = __shfl_sync(0xFFFFFFFF, maxval, 0);
    sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);

#pragma unroll
    for (int i = laneId; i < C; i += warpSize)
        y[i] = expf(x[i] - maxval) / sumval;
}




extern "C" void *softmax(int thread_id, Tensor *tensor)
{
  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  
  dims =  format_LinearLayer_Dims(dims);

  int B = dims[0];
  int C = dims[1];


  int grid_size, block_size, shared_mem_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];


  tensor->Sync();
  float *probs = get_from_pool(thread_id, B*C, "softmax");
  cudaStream_t stream = ThreadsStream[thread_id];
  set_to_zero_kernel<<<grid_size, block_size, 0, stream>>>(probs, B*C);


  
  /*
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size  = B;
  block_size = grid_block_mem_sizes[1];

  shared_mem_size = 2 * block_size / 32 * sizeof(float);
  softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_ptr, probs, B, C);
  */
 
 
  grid_block_mem_sizes = CalculateSimpleWarpGridAndBlockSizes(B);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  online_softmax<<<grid_size, block_size, 0, stream>>>(tensor_ptr, probs, B, C);



  Tensor *new_tensor = createTensor(probs, tensor->dims, tensor->dims_prod, false, "");
  new_tensor->op=softmax_op;
  return new_tensor;
}




extern "C" void *self_attn(int thread_id, Tensor *tensor)
{
  float *tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  
  dims =  format_LinearLayer_Dims(dims);

  int B = dims[0];
  int C = dims[1];


  int grid_size, block_size, shared_mem_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];


  tensor->Sync();
  float *probs = get_from_pool(thread_id, B*C, "self_attn");
  cudaStream_t stream = ThreadsStream[thread_id];
  set_to_zero_kernel<<<grid_size, block_size, 0, stream>>>(probs, B*C);



  grid_block_mem_sizes = CalculateSimpleWarpGridAndBlockSizes(B);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];


  online_softmax<<<grid_size, block_size, 0, stream>>>(tensor_ptr, probs, B, C);
  
  
  Tensor *new_tensor = createTensor(probs, dims, tensor->dims_prod, false, "");
  new_tensor->op=self_attn_op;
  return new_tensor;

}



__global__ void gather_last_dim_kernel(float* y, const float* tensor, const float *tensor_idx,
                                      const int leading_dim, float dims_prod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < dims_prod) {
      int t_idx = (int)tensor_idx[idx];
      y[idx] = tensor[idx*leading_dim + t_idx];
    }
}



extern "C" void *gather(int thread_id, Tensor *tensor, Tensor *idx_tensor, float dim)
{
  //std::cout << "Gather THREAD IS: " << thread_id << "\n";


  if(dim<0)
    dim = tensor->dims.size()+dim;

  if(dim == tensor->dims.size()-1)
  {
    //std::cout << "Gather over last dim"  << "\n";

    float *tensor_ptr = tensor->tensor_ptr;
    std::vector<float> dims, new_dims;
    dims = tensor->dims;
    new_dims = RemoveLastDim(dims);
    float leading_dim = dims[dim];

    //PrintDims(dims);
    //PrintDims(new_dims);

    
    float dims_prod = tensor->dims_prod;
    float new_dims_prod = DimsProd(new_dims);

    int grid_size, block_size, shared_mem_size; 
    std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(new_dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];
    

    float *y = get_from_pool(thread_id, new_dims_prod, "gather");
    //float *y;
    

    tensor->Sync();
    cudaStream_t stream = ThreadsStream[thread_id];
    gather_last_dim_kernel<<<grid_size, block_size, 0, stream>>>(y, tensor->tensor_ptr, idx_tensor->tensor_ptr, leading_dim, new_dims_prod);



    

    Tensor *new_tensor = createTensor(y, new_dims, new_dims_prod, false, "");
    //idx_tensor->op = detach_op;
    new_tensor->AttrNodes(tensor, wrapTensorWithDetached(idx_tensor), gather_last_dim_op);
    //new_tensor->AttrLNode(idx_tensor, gather_last_dim_op);
    todo_backward_tensors.push_back(new_tensor);
    return new_tensor;
  }
}

__global__ void gather_last_dim_backward_kernel(float* dx, const float* dy, const float *tensor_idx,
                                      const int leading_dim, float dims_prod) {
    // Handles idx repetition
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < dims_prod) {
      int t_idx = (int)tensor_idx[idx];

      float *indexed_dx = dx + idx*leading_dim + t_idx;

      atomicAdd(indexed_dx, dy[idx]);
      //dx[idx*leading_dim + t_idx] = dy[idx];
    }
}

void gather_last_dim_backward(float *dx, float *dy, Tensor *node)
{
  // consider dx was set to zero already
  

  float *idx = node->R_Node->tensor_ptr;

  std::vector<float> dims = node->L_Node->dims;
  int leading_dim = dims[dims.size()-1];

  float dims_prod = node->dims_prod;


  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];


  gather_last_dim_backward_kernel<<<grid_size, block_size, 0, main_stream>>>(dx, dy, idx, leading_dim, dims_prod);

  //PrintTensorF(idx, 1, node->R_Node->dims_prod);
  //PrintTensorF(dx, dims[0], dims[1]);

}






__global__ void rl_discounted_return_kernel(float *G, const float *rewards, const float *terminated,
                                      const int T, const float gamma, const float dims_prod) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    float g=0;
    if (b < dims_prod) {
        for(int t=T-1; t>=0; t--)
        {
          g += rewards[b*T+t] * powf(gamma, t);
          g = g * (1-terminated[b*T+t]);
        }
        G[b] = g;
    }
}


extern "C" void *rl_discounted_return(int thread_id, Tensor *reward, Tensor *terminated, float gamma)
{
  //std::cout << "rl_discounted_return THREAD IS: " << thread_id << "\n";

  std::vector<float> dims = reward->dims;

  if (reward->dims.size()!=2||terminated->dims.size()!=2)
    LogErrorS("rl_discounted_return requires dims [B, T]");

  int B = dims[0];
  int T = dims[1];
  

  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(B);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  

  float *G = get_from_pool(thread_id, B, "rl_discounted_return");

  reward->Sync();
  terminated->Sync();

  cudaStream_t stream = ThreadsStream[thread_id];
  rl_discounted_return_kernel<<<grid_size, block_size, 0, stream>>>(G, reward->tensor_ptr, terminated->tensor_ptr, T, gamma, B);



  Tensor *new_tensor = createTensor(G, {(float)B}, B, false, "");
  new_tensor->AttrNodes(reward, terminated, detach_op);
  return new_tensor;
}





class BatchNorm2d
{
  public:
    cudnnTensorDescriptor_t input_desc, output_desc, scale_bias_mean_var_desc;
    
    float* scale=nullptr;
    float* bias=nullptr;
    float* running_mean=nullptr;
    float* running_var=nullptr;
    float* saved_mean=nullptr;
    float* saved_var=nullptr;
    float* dscale, dbias;
    int B = 0;
    int C;
    int H = 0;
    int W = 0;
    std::string Name;

    BatchNorm2d(int C, std::string Name)
        : C(C), Name(Name) {
      NamedTensorsT[Name] = new Tensor();
      NamedTensorsT[Name+"_bias"] = new Tensor();
    }

  
  void SetDescriptors(int, int, int, Tensor *);
  void InitMovingAverages();
  float *Forward(Tensor *, int, int, int, int, int);
  void Backward(float *, float *, float *, float *, float *);

};


class Conv2d
{
  public:
    // Forward
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnConvolutionFwdAlgo_t fwd_algo;  

    // Weight backward grad
    cudnnConvolutionBwdFilterAlgo_t w_bwd_algo;
    // Input backward grad
    cudnnConvolutionBwdDataAlgo_t y_bwd_algo;

    std::size_t workspace_size, workspace_size_w_back, workspace_size_y_back;
    float *d_workspace, *d_workspace_w_back, *d_workspace_y_back;


    float* d_filter=nullptr;
    float* d_filter_g=nullptr;
    int C, OC, ks, stride, padding, out_H, out_W;
    int B = 0;
    int H = 0;
    int W = 0;
    std::string Init, Name;

    Conv2d(int C, int OC, int ks, int stride, int padding, std::string Init, std::string Name) 
        : C(C), OC(OC), ks(ks), stride(stride), padding(padding), Init(Init), Name(Name) {
      NamedTensorsT[Name] = new Tensor();
      d_filter=nullptr;
      d_workspace=nullptr;
      d_workspace_w_back=nullptr;
      d_workspace_y_back=nullptr;
      workspace_size=0;
      workspace_size_w_back=0;
      workspace_size_y_back=0;
    }

  


  void SetDescriptors(int, int, int, Tensor *tensor);
  void InitFilters();
  float *Forward(Tensor *, int, int, int, int);
  void Backward(float *, float *, float *, float *);

};



class BN2dRelu
{
  public:
    cudnnTensorDescriptor_t input_desc, output_desc, intermediate_desc, scale_bias_mean_var_desc;
    cudnnActivationDescriptor_t activation_desc;

    float* scale=nullptr;
    float* bias=nullptr;
    float* running_mean=nullptr;
    float* running_var=nullptr;
    float* saved_mean=nullptr;
    float* saved_var=nullptr;
    float* dscale, dbias;
    int B = 0;
    int C;
    int H = 0;
    int W = 0;
    std::string Name;

    BN2dRelu(int C, std::string Name)
        : C(C), Name(Name) {
      NamedTensorsT[Name] = new Tensor();
      NamedTensorsT[Name+"_bias"] = new Tensor();
    }

  
  void SetDescriptors(int, int, int, Tensor *);
  void InitMovingAverages();
  float *Forward(Tensor *, int, int, int, int, int);
  void Backward(float *, float *, float *, float *, float *, float *, float *, float *);

};


class Relu
{
  public:
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnActivationDescriptor_t activation_desc;


    int B = 0;
    int C = 0;
    int H = 0;
    int W = 0;
    std::string Name;

    Relu(std::string Name)
        : Name(Name) {}

  
  void SetDescriptors(int, int, int, int, Tensor *);
  float *Forward(Tensor *, int, int, int, int);
  void Backward(float *, float *, float *, float *);

};


class MaxPool2d
{
  public:
    // Forward
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnPoolingDescriptor_t pooling_desc;
    
    std::string Type;
    int ks, stride, padding, out_H, out_W;
    int B = 0;
    int C = 0;
    int H = 0;
    int W = 0;

    MaxPool2d(int ks, int stride, int padding, std::string Type)
        : ks(ks), stride(stride), padding(padding), Type(Type) {}

  


  void SetDescriptors(int, int, int, int, Tensor *);
  float *Forward(Tensor *, int, int, int, int, int);
  void Backward(float *, float *, float *, float *);

};



class Embedding
{
  public:
    
    int C, OC, B;
    std::string Init, Name;
    float *W, *dW;
    bool changed_descriptors;

    Embedding(int C, int OC, std::string Init, std::string Name)
        : C(C), OC(OC), Init(Init), Name(Name) {
      // C == num_codebooks
      B = 0;

      float *w_cpu;
          
      //w_cpu = make_xavier_uniform_float(OC*C, OC,  C);
      //w_cpu = make_normal(OC*C);
      w_cpu = make_uniform(OC*C);


      
      W = get_from_pool(0, OC*C, "Embedding W");
      cudaMemcpy(W, w_cpu, OC*C*sizeof(float), cudaMemcpyHostToDevice);

      Tensor *tensor_W = createTensor(W, {(float)C,(float)OC}, OC*C, true, Name);
      
      
      


      dW = get_from_pool(0, OC*C, "embedding dW");
      set_to_zero_kernel<<<std::ceil((OC*C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dW, OC*C);

      NamedTensorsT[Name] = tensor_W;
      NamedParamGrads[Name] = dW;

      delete[] w_cpu;

      changed_descriptors=false;
    }
  
  void SetDescriptors(int);
  void SetBackwardDescriptors();
  float *Forward(Tensor *, int, int);
  void Backward(float *, float *);
};


class LSTM
{
  public:
    
    int C, OC, B, T;
    std::string Init, Name;
    float *W, *U, *x_out, *fused_out, *b, *all_ht, *all_ct, *dW, *dU, *dB, *d_ht, *d_ct, *d_ifoc;
    bool changed_descriptors, first_backward;

    LSTM(int C, int OC, std::string Init, std::string Name)
        : C(C), OC(OC), Init(Init), Name(Name) {
      B = 0;
      T = 0;

      x_out = nullptr;
      fused_out = nullptr;

      float *w_cpu, *u_cpu, *b_cpu;
      w_cpu = make_N_orthogonals(4, OC, OC);
      //w_cpu = make_lstm_init_xavier(OC, OC);
      u_cpu = make_lstm_init_xavier(OC,  C);
      
      //w_cpu = make_lstm_torch(OC, OC);
      //u_cpu = make_lstm_torch(OC, C);

      b_cpu = make_lstm_bias(OC);


      cudaMalloc(&W, 4*OC*OC*sizeof(float));
      cudaMalloc(&U, 4*OC* C*sizeof(float));
      cudaMalloc(&b, 4*OC*   sizeof(float));
      cudaMemcpy(W, w_cpu, 4*OC*OC*sizeof(float), cudaMemcpyHostToDevice); // ht weight
      cudaMemcpy(U, u_cpu, 4*OC* C*sizeof(float), cudaMemcpyHostToDevice); // x weight
      cudaMemcpy(b, b_cpu, 4*OC*   sizeof(float), cudaMemcpyHostToDevice); // bias
 
      Tensor *tensor_W = createTensor(W, {4*(float)OC, (float)OC}, 4*OC*OC, true, Name+"W");
      Tensor *tensor_U = createTensor(U, {4*(float)OC, (float)C},  4*OC* C, true, Name+"U");
      Tensor *tensor_B = createTensor(b, {4*(float)OC},            4*OC   , true, Name+"b");
      tensor_W->SetIsWeight();
      tensor_U->SetIsWeight();
      

      NamedTensorsT[Name+"W"] = tensor_W;
      NamedTensorsT[Name+"U"] = tensor_U;
      NamedTensorsT[Name+"B"] = tensor_B;

      delete[] w_cpu;
      delete[] u_cpu;

      changed_descriptors = false;
      first_backward = true;
    }

  
  void SetDescriptors(int, int, int);
  void SetBackwardDescriptors();
  void FirstBackward();
  float *Forward(Tensor *, Tensor *, Tensor *, int, int, int);
  void Backward(float *, float *, float *);

};



class Linear
{
  public:
    
    int B, C, OC;
    std::string Init, Name;
    float *W, *dW;
    bool first_backward, changed_descriptors;

    int_vec *Notators;
    bool _fp32;

    Linear(int C, int OC, std::string Init, int_vec *Notators, std::string Name)
        : C(C), OC(OC), Init(Init), Notators(Notators), Name(Name) {
      B = 0;

      _fp32 = true;
      if (in_int_ptr(fp16, Notators->vec, Notators->size))
        _fp32 = false;



      
      float *W_cpu;
      int product = OC*C;


      if (Init=="randu")
        W_cpu = make_random_float_uniform(product);
      if (Init=="zeros")
        W_cpu = make_zeros_float(product);
      if (Init=="ones")
        W_cpu = make_ones_float(product);
      if (Init=="normal")
        W_cpu = make_normal(product);
      if (Init=="xavu")
        W_cpu = make_xavier_uniform_float(product, C, OC);
      if (Init=="xavu_relu")
        W_cpu = make_xavier_uniform_float_relu(product, C, OC);
      if (Init=="xavu_tanh")
        W_cpu = make_xavier_uniform_float_tanh(product, C, OC);
      if (Init=="he_normal_relu")
        W_cpu = make_he_normal_float_relu(product, C);
      if (Init=="init_gpt")
        W_cpu = make_gpt_init(product);
      if (Init=="int")
        W_cpu = make_random_int(product, 10);
      if (Init=="binary")
        W_cpu = make_random_int(product, 1);


      cudaMalloc(&W,       product * sizeof(float));
      cudaMemcpy(W, W_cpu, product * sizeof(float), cudaMemcpyHostToDevice);

      Tensor *tensor_W = createTensor(W, {(float)OC*(float)C}, product, true, Name+"W");
      tensor_W->SetIsWeight();

      NamedTensorsT[Name+"W"] = tensor_W;

      delete[] W_cpu;
      delete[] Notators;


      first_backward = true;
      changed_descriptors = false;
    }
  
  float *Forward(Tensor *, int);
  void SetDescriptors(int, int);
  void Backward(float *, float *, float *);
  void SetBackwardDescriptors();
  void FirstBackward();
};




class MHSA
{
  public:
    
    int B, T, maxT, nh, C, d, B_back, T_back;
    int M, Br, Bc, Tr, Tc;
    int Br_back, Bc_back, Tr_back, Tc_back;
    std::string Init, Name;
    float *W, *W_proj, *l, *qkv, *out, *qkv_back, *out_back, *l_back, *dW, *dW_proj;
    bool first_backward, changed_descriptors, _fp32, _fp32_back, _causal;
    int_vec *Notators;

    MHSA(int nh, int C, int maxT, std::string Init, int_vec *Notators, std::string Name)
        : nh(nh), C(C), maxT(maxT), Init(Init), Notators(Notators), Name(Name) {
      B = 0;
      T = 0;
      d = C/nh;
      M = deviceProp.sharedMemPerBlock;


      _fp32 = true;
      _fp32_back = true;
      _causal = false;

      if (in_int_ptr(fp16, Notators->vec, Notators->size))
      {
        _fp32 = false;
        _fp32_back = false;
      }
      if (in_int_ptr(causal, Notators->vec, Notators->size))
        _causal = true;

      
      float *W_cpu, *W_proj_cpu;

      //W_cpu = make_gpt_init(3*C*C);
      //W_proj_cpu = make_gpt_init(C*C);
      W_cpu = make_xavier_uniform_float(3*C*C, C, 3*C);
      W_proj_cpu = make_xavier_uniform_float(C*C, C, C);

      cudaMalloc(&W,       3*C*C*sizeof(float));
      cudaMalloc(&W_proj,  C*C*sizeof(float));
      cudaMemcpy(W, W_cpu, 3*C*C * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(W_proj, W_proj_cpu, C*C * sizeof(float), cudaMemcpyHostToDevice);

      Tensor *tensor_W = createTensor(W, {3*(float)C*(float)C}, 3*C*C, true, Name+"W");
      Tensor *tensor_W_proj = createTensor(W_proj, {(float)C*(float)C}, C*C, true, Name+"W_proj");
      tensor_W->SetIsWeight();
      tensor_W_proj->SetIsWeight();

      NamedTensorsT[Name+"W"] = tensor_W;
      NamedTensorsT[Name+"W_proj"] = tensor_W_proj;

      delete[] W_cpu;
      delete[] W_proj_cpu;
      delete[] Notators;


      first_backward = true;
      changed_descriptors = false;
    }
  
  float *Forward(Tensor *, int, int, int);
  void SetDescriptors(int, int, int);
  void Backward(float *, float *, float *);
  void SetBackwardDescriptors();
  void FirstBackward();
};



//global
static std::map<std::string, std::unique_ptr<BN2dRelu>> NamedBN2dRelu;
static std::map<std::string, std::unique_ptr<Relu>> NamedRelu;
static std::map<std::string, std::unique_ptr<MaxPool2d>> NamedMaxPool2d;
static std::map<std::string, std::unique_ptr<Conv2d>> NamedConv2d;
static std::map<std::string, std::unique_ptr<BatchNorm2d>> NamedBatchNorm2d;
static std::map<std::string, std::unique_ptr<Embedding>> NamedEmbedding;
static std::map<std::string, std::unique_ptr<LSTM>> NamedLSTM;
static std::map<std::string, std::unique_ptr<Linear>> NamedLinear;
static std::map<std::string, std::unique_ptr<MHSA>> NamedMHSA;




__device__ float _truncf(float value) {
    float factor = 10000.0f;  // 10^4 for four decimal places
    return truncf(value * factor) / factor;
}


__global__ void flash_attn_kernel(float *o, const float *qkv, float *l,
                                  const int B, const int nh, const int T, const int d, const int C, const float d_scale, const int Bc, const int Br,
                                  const int Tc, const int Tr, const int tile_size, const float warps_per_block, const int threads_per_block)
{
  int b = blockIdx.y; // batch idx
  int h = blockIdx.x; // head  idx

  if(b>=B||h>=nh)
    return;


  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int tid = (ty * blockDim.x + tx);
  
  int warpId = tid / warpSize;
  int laneId = tid % warpSize;

  
  extern __shared__ float smem[];
  

  float *q_smem        = smem;                               // [Br,  d]
  float *k_smem        = smem + Br*d;                        // [Bc,  d]
  float *v_smem        = smem + (Br+Bc)*d;                   // [Bc,  d]
  float *o_smem        = smem + (Br+2*Bc)*d;                 // [Br,  d]
  float *Sij_smem      = smem + 2*(Br+Bc)*d;                 // [Br, Bc]
  float *l_smem        = smem + 2*(Br+Bc)*d + Br*Bc;         // [Br]
  float *m_smem        = smem + 2*(Br+Bc)*d + Br*Bc + Br;    // [Br]
  float *last_m_smem   = smem + 2*(Br+Bc)*d + Br*Bc + 2*Br;  // [Br]
  
  
  const float *q = qkv;
  const float *k = qkv;
  const float *v = qkv;



  for (int i=0; i<Tr; ++i)
  {

    // Load Qi and Oi of size [Br, d] each
    
    for (int tile=0; tile<ceilf((Br*d)/(float)threads_per_block); ++tile)
    {
      int tile_idx = tile*threads_per_block + tid;
      int br = tile_idx / d;
      int _d = tile_idx % d;


      if(br<Br)
      {
        l_smem[br] = 0;
        m_smem[br] = -INFINITY;
        last_m_smem[br] = -INFINITY;

        if((i*Br+br)<T)
          q_smem[br*d + _d] = q[b*T*3*C + (i*Br+br)*3*C + h*d + _d];
        else
          q_smem[br*d + _d] = 0;
        
        o_smem[br*d + _d] = 0;
      }
      
    }


      
    for (int j=0; j<Tc; ++j)
    {


      // Load Kj and Vj of size [Bc, d] each
      for (int tile=0; tile<ceilf((Bc*d)/(float)threads_per_block); ++tile)
      {
        int tile_idx = tile*threads_per_block + tid;
        int bc = tile_idx / d;
        int _d = tile_idx % d;

        
        
        if(bc<Bc)
        {
          if((j*Bc+bc)<T)
          {
            k_smem[bc*d + _d] = k[b*T*3*C + (j*Bc+bc)*3*C +   C + h*d + _d];
            v_smem[bc*d + _d] = v[b*T*3*C + (j*Bc+bc)*3*C + 2*C + h*d + _d];
          } else {
            k_smem[bc*d + _d] = 0;
            v_smem[bc*d + _d] = 0;
          }
        }
        
      }
      __syncthreads();



      // compute q @ k.T
      for (int warp_tile=0; warp_tile < std::ceil((Br*Bc)/(float)warps_per_block); ++warp_tile)
      {
        int wid = warp_tile * warps_per_block + warpId;
        int br = wid / Bc; 
        int bc = wid % Bc;


        if (br<Br)
          Sij_smem[br*Bc + bc] = 0.0f;

        // \sum_i q[i] @ k[i].T  for each lane
        
        
        float sij = 0;
        for (int lane_tile = laneId; lane_tile < d; lane_tile += warpSize)
        {
          if (bc<Bc && br<Br && (j*Bc+bc)<T && (i*Br+br)<T)
            sij += q_smem[br*d + lane_tile]*k_smem[bc*d + lane_tile];
        }

        

        // \sum_i q[i] @ k[i].T  across the warp
        float mask_sij;
        for (int mask = warpSize/2; mask>0; mask>>=1)
        {
          __syncwarp();
          mask_sij = __shfl_down_sync(0xFFFFFFFF, sij, mask);
          sij += mask_sij;
        }
        sij = __shfl_sync(0xFFFFFFFF, sij, 0);


        if (bc<Bc && br<Br && (j*Bc+bc)<T && (i*Br+br)<T && laneId==0)
          Sij_smem[br*Bc + bc] = sij/d_scale;
      }
      __syncthreads();



      ///---///

      

      // get the softmax statistics and Pij
      for (int warp_tile=0; warp_tile < std::ceil(Br/warps_per_block); ++warp_tile)
      {
        int br = warp_tile * warps_per_block + warpId;
        
        
        float maxval = -INFINITY;
        if (br<Br && (i*Br+br)<T)
        {
          
          for (int lane_tile=laneId; lane_tile<Bc && (j*Bc+lane_tile)<T; lane_tile+=warpSize)
            maxval = fmaxf(maxval, Sij_smem[br*Bc + lane_tile]);
          
          
          

          float mask_maxval;
          for (int mask=warpSize/2; mask>0; mask>>=1)
          {
            __syncwarp();
            mask_maxval = __shfl_down_sync(0xFFFFFFFF, maxval, mask);

            if (mask_maxval > maxval)
                maxval = mask_maxval;
            
          }
          maxval = __shfl_sync(0xFFFFFFFF, maxval, 0);

        }



        if(laneId==0 && br<Br && (i*Br+br)<T)
          m_smem[br] = fmaxf(last_m_smem[br], maxval);
        
        
        

        

        // Pij = exp(Sij - mi)
        if (br<Br&&(i*Br+br)<T)
        {
          for (int lane_tile=laneId; lane_tile<Bc && (j*Bc+lane_tile)<T; lane_tile+=warpSize)
            Sij_smem[br*Bc + lane_tile] = expf(Sij_smem[br*Bc + lane_tile] - m_smem[br]);
        }
        
        
        
        float sumval=0.0f;
        if (br<Br&&(i*Br+br)<T)
        {
          
          for (int lane_tile=laneId; lane_tile<Bc && (j*Bc+lane_tile)<T; lane_tile+=warpSize)
            sumval += Sij_smem[br*Bc + lane_tile];
          
          

          float mask_sumval;
          for (int mask=warpSize/2; mask>0; mask>>=1)
          {
            __syncwarp();
            mask_sumval = __shfl_down_sync(0xFFFFFFFF, sumval, mask);
            sumval += mask_sumval;
          }
          sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);
        }

        if(laneId==0 && br<Br && (i*Br+br)<T)
        {
          if(j==0)
            l_smem[br] = sumval;
          else
            l_smem[br] = expf(last_m_smem[br]-m_smem[br])*l_smem[br] + sumval;
            //l_smem[br] = expf(last_m_smem[br]-m_smem[br])*l_smem[br] + expf(maxval-m_smem[br])*sumval;
        }

      }

      __syncthreads();

      


      for (int warp_tile=0; warp_tile < std::ceil((Br*d)/warps_per_block); ++warp_tile)
      {
        int wid = warp_tile * warps_per_block + warpId;
        int br = wid / d;
        int _d = wid % d;


        float pv=0;

        if(br<Br && (i*Br+br)<T)
        {
          
          for (int lane_tile=laneId; lane_tile<Bc && (j*Bc+lane_tile)<T; lane_tile+=warpSize)
            pv += Sij_smem[br*Bc + lane_tile] * v_smem[lane_tile*d + _d];
          
          

          float mask_p;
          for (int mask=warpSize/2; mask>0; mask>>=1)
          {
            __syncwarp();
            mask_p = __shfl_down_sync(0xFFFFFFFF, pv, mask);
            pv+=mask_p;
          }
          pv = __shfl_sync(0xFFFFFFFF, pv, 0);


          
          if (laneId==0)
          { 
            if (j==0)
              o_smem[br*d + _d] = pv;
            else
              o_smem[br*d + _d] = o_smem[br*d + _d]*expf(last_m_smem[br] - m_smem[br]) + pv;
          }
        }
        
      }

      __syncthreads();




      for (int warp_tile=0; warp_tile < std::ceil(Br/warps_per_block); ++warp_tile)
      {
        int br = warp_tile*warps_per_block + warpId;
        
        if (br<Br && (i*Br+br)<T && laneId==0)
          last_m_smem[br] = m_smem[br];
        
      }
      __syncthreads();
    }
    





    // Load Oi into HBM O
    for (int warp_tile=0; warp_tile < std::ceil(Br/(float)warps_per_block); ++warp_tile)
    {
      int br = warp_tile*warps_per_block + warpId;
      
      
      
      for (int lane_tile=laneId; lane_tile<d; lane_tile+=warpSize)
      {
        if (br<Br && (i*Br+br)<T)
          o_smem[br*d + lane_tile] = o_smem[br*d + lane_tile]/l_smem[br];
      }
      
      

      __syncthreads();


      if ((i*Br+br)<T && br<Br)
      {
        for (int lane_tile=laneId; lane_tile<d; lane_tile+=warpSize)
          o[b*T*C + (i*Br+br)*C + h*d + lane_tile] = o_smem[br*d + lane_tile];
        
        if (laneId==0)
        {

          l[b*T*nh + (i*Br+br)*nh + h] = m_smem[br] + logf(l_smem[br]);
          //l[b*T*nh + (i*Br+br)*nh + h] = l_smem[br];
          //m[b*T*nh + (i*Br+br)*nh + h] = m_smem[br];
        }
      }
      
      __syncthreads();
    }
  }
}










template<int WMMA_T, int num_warps>
__global__ void flash_attn_fp16_kernel(float *o, const float *qkv, float *l,
                                  const int B, const int nh, const int T, const int d, const int C, const float d_scale, const int Bc, const int Br,
                                  const int Tc, const int Tr, const int tile_size, const float warps_per_block, const int threads_per_block)
{
  int b = blockIdx.y; // batch idx
  int h = blockIdx.x; // head  idx

  if(b>=B||h>=nh)
    return;


  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int tid = (ty * blockDim.x + tx);
  
  int warpId = tid / warpSize;
  int laneId = tid % warpSize;

  int mw = laneId / 4; // up to 8
  int ml = laneId % 4; // up to 4

  int warp_y = warpId / 4; 
  int warp_x = warpId % 4;

  
  extern __shared__ float smem[];
  

  float *q_smem        = smem;                               // [Br,  d]
  float *k_smem        = smem + Br*d;                        // [Bc,  d]
  float *v_smem        = smem + (Br+Bc)*d;                   // [Bc,  d]
  float *o_smem        = smem + (Br+2*Bc)*d;                 // [Br,  d]
  float *Sij_smem      = smem + 2*(Br+Bc)*d;                 // [Br, Bc]
  float *l_smem        = smem + 2*(Br+Bc)*d + 32*32;         // [Br]
  float *m_smem        = smem + 2*(Br+Bc)*d + 32*32 + Br;    // [Br]
  float *last_m_smem   = smem + 2*(Br+Bc)*d + 32*32 + 2*Br;  // [Br]
  
  
  const float *q = qkv;
  const float *k = qkv;
  const float *v = qkv;



  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> v_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> Sij_frag;

  




  // smem xor shuffle auxiliars
  int xor_addr = smem_xor_cp_async(laneId);
  // int B_row = warpId*4 + ml; // Limit sizes of Bc and Br are 32, and 64 for B_row with 16 warps. Thus, there is no need to loop-tile over rows.
  float K_tile = 8*4; // mw_lanes  x  cp_async size
  
  int br = warp_y*WMMA_T+warp_x*4+ml;
  int bc = warp_x*WMMA_T+warp_y*4+ml;

  int k_count=0;
  int k_stride;


  for (int i=0; i<Tr; ++i)
  {
    // Load Qi and Oi of size [Br, d] each

    
    for (int tile=0; tile<d; tile+=K_tile)
    {


      if(br<Br)
      {
        l_smem[br] = 0;
        m_smem[br] = -INFINITY;
        last_m_smem[br] = -INFINITY;

        // if((i*Br+br)<T)
        //   q_smem[br*d + _d] = q[b*T*3*C + (i*Br+br)*3*C + h*d + _d];
        // else
        //   q_smem[br*d + _d] = 0;
        
        // o_smem[br*d + _d] = 0;
      }

      if ((warp_y*4+warp_x)<Br && (warp_x*4+ml)<WMMA_T)
      {
        float const *gmem_ptr = q + b*T*3*C + (i*Br+br)*3*C + h*d + tile+mw*4;//(warpY*WMMA_T+row_aux1)*C + tile+mw*4;
        
        // extra *2 to accomodate 32 instead of 16 C (i.e, the whole warpSize)
        //       *4 is necessary as it needs to accomodate 4 consecutive floats
        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&q_smem[(warp_y*4 + warp_x)*d + tile*4 + xor_addr]);

        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
                      :: "r"(smem_int_ptr),
                        "l"(gmem_ptr),
                        "n"(16),
                        "r"(((i*Br+br)<T&&br<Br) ? std::min((( C-(tile+mw*4)) /4)*4, 16) : 0)); // incorrect 0 padding yet



        gmem_ptr = o + b*T*C + (i*Br+br)*C + h*d + tile+mw*4;
        smem_int_ptr = cast_smem_ptr_to_uint(&o_smem[(warp_y*4 + warp_x)*d + tile*4 + xor_addr]);
        
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
                      :: "r"(smem_int_ptr),
                        "l"(gmem_ptr),
                        "n"(16),
                        "r"(0)); // Set Oi to 0
      }
    }


    
    for (int j=0; j<Tc; ++j)
    {


      // Load Kj and Vj of size [Bc, d] each
      for (int tile=0; tile<d; tile+=K_tile)
      {
        
        if ((warp_y*4+warp_x)<Br && (warp_y*4+ml)<WMMA_T && bc<Bc)
        {
          float const *gmem_ptr = k + b*T*3*C + (j*Bc+bc)*3*C + C + h*d + tile+mw*4;
          uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&k_smem[(warp_x*4 + warp_y)*d + tile*4 + xor_addr]);


          asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
                        :: "r"(smem_int_ptr),
                            "l"(gmem_ptr),
                            "n"(16),
                            "r"(((j*Bc+bc)<T&&bc<Bc) ? std::min((( C-(tile+mw*4)) /4)*4, 16) : 0)); // incorrect 0 padding yet


          // gmem_ptr = v + b*T*3*C + (j*Bc+bc)*3*C + 2*C + h*d + tile+mw*4;
          // smem_int_ptr = cast_smem_ptr_to_uint(&v_smem[(warp_x*4 + warp_y)*d + tile*4 + xor_addr]);

          // asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
          //               :: "r"(smem_int_ptr),
          //                   "l"(gmem_ptr),
          //                   "n"(16),
          //                   "r"(((j*Bc+bc)<T&&bc<Bc) ? std::min((( C-(tile+mw*4)) /4)*4, 16) : 0)); // incorrect 0 padding yet        
        }
      }

      for (int tile=0; tile<ceilf((Bc*d)/(float)threads_per_block); ++tile)
      {
        int tile_idx = tile*threads_per_block + tid;
        int bc = tile_idx / d;
        int _d = tile_idx % d;

        
        
        if(bc<Bc)
        {
          if((j*Bc+bc)<T)
          {
            v_smem[bc*d + _d] = v[b*T*3*C + (j*Bc+bc)*3*C + 2*C + h*d + _d];
          } else {
            v_smem[bc*d + _d] = 0;
          }
        }
        
      }

      asm volatile("cp.async.wait_all;");
      __syncthreads();



      // compute q @ k.T
      
      k_count=0;

      int _br = warp_y*WMMA_T/4;
      int _bc = warp_x*WMMA_T/4;

      wmma::fill_fragment(Sij_frag, 0.0f);

      for (int tile=0; tile < d; tile += WMMA_T)
      {
        k_stride=k_count%2;
        k_count++;

        int _tile = (tile/32)*32;


        if (_br<Br&&_bc<Bc)
        {
          
          const auto func_q = [&](const unsigned* frag_index_list,
              const unsigned fragment_index_count,
              const unsigned i,
              const unsigned j) {

              int wi = i/4;
              int xi = i%4;

              int xj = j/4;
              int wj = j%4;

              int offset = smem_dexor_from_cp_async(xi, xj*2 + k_stride)+wj;

              __half tmp = __float2half(*(q_smem + (_br+ wi)*d + _tile*4 + offset));
      #pragma unroll
              for (unsigned f = 0; f < fragment_index_count; f++)
                  q_frag.x[frag_index_list[f]] = tmp;
          };
          __syncwarp();
          wmma_foreach_ij(
            q_frag,
            func_q
          );


        
          const auto func_k = [&](const unsigned* frag_index_list,
                const unsigned fragment_index_count,
                const unsigned i,
                const unsigned j) {
              

                int wj = j/4;
                int xj = j%4;
              
                int xi = i/4;
                int wi = i%4;


                int offset = smem_dexor_from_cp_async(xj, xi*2+k_stride)+wi;

              
                __half tmp = __float2half(*(k_smem + (_bc+wj)*d + _tile*4 + offset));
        #pragma unroll
                for (unsigned f = 0; f < fragment_index_count; f++)
                  k_frag.x[frag_index_list[f]] = tmp;
            };

          __syncwarp();
          wmma_foreach_ij(
            k_frag,
            func_k
          );
          

          wmma::mma_sync(Sij_frag, q_frag, k_frag, Sij_frag);
        }
      }



      if (_br<Br&&_bc<Bc)
      {
        const auto func_sij = [&](const unsigned* frag_index_list,
        const unsigned fragment_index_count,
        const unsigned i,
        const unsigned j) {


          // if((warpId+h+b)==0&&laneId==1)
          // {
          //   printf("%d - %d\n", i, j);
          // }

          Sij_smem[(_br+i)*Bc + _bc+j] = Sij_frag.x[frag_index_list[0]]/d_scale;
            
        };

        __syncwarp();
        wmma_foreach_ij(
          Sij_frag,
          func_sij
        );

        
      }

    
      // if((warpId+laneId+h+b)==0)
      // { 
      //   for (int i=0; i<Br*Bc; ++i)
      //   {
      //     printf("%f, ", Sij_smem[i]);
      //     if ((i+1)%Bc==0)
      //       printf("\n");
      //   }
      //   printf("\n\n");
      // }




      /*
      for (int warp_tile=0; warp_tile < std::ceil((Br*Bc)/(float)warps_per_block); ++warp_tile)
      {
        k_stride=k_count%2;
        k_count++;
        
        int wid = warp_tile * warps_per_block + warpId;
        int _br = wid / Bc; 
        int _bc = wid % Bc;


        if (br<Br)
          Sij_smem[_br*Bc+_bc]=0;

        // \sum_i q[i] @ k[i].T  for each lane
        float sij = 0;
        for (int lane_tile = laneId; lane_tile < d; lane_tile += warpSize)
        {
          if (bc<Bc && br<Br && (j*Bc+bc)<T && (i*Br+br)<T)
            sij += q_smem[br*d + lane_tile]*k_smem[bc*d + lane_tile];
        }
        

        // \sum_i q[i] @ k[i].T  across the warp
        float mask_sij;
        for (int mask = warpSize/2; mask>0; mask>>=1)
        {
          __syncwarp();
          mask_sij = __shfl_down_sync(0xFFFFFFFF, sij, mask);
          sij += mask_sij;
        }
        sij = __shfl_sync(0xFFFFFFFF, sij, 0);


        if (bc<Bc && br<Br && (j*Bc+bc)<T && (i*Br+br)<T && laneId==0)
          Sij_smem[br*Bc + bc] = sij/d_scale;
      }
      */
      __syncthreads();



      ///---///

      

      // get the softmax statistics and Pij
      for (int warp_tile=0; warp_tile < std::ceil(Br/warps_per_block); ++warp_tile)
      {
        int br = warp_tile * warps_per_block + warpId;
        
        
        float maxval = -INFINITY;
        if (br<Br && (i*Br+br)<T)
        {
          
          for (int lane_tile=laneId; lane_tile<Bc && (j*Bc+lane_tile)<T; lane_tile+=warpSize)
            maxval = fmaxf(maxval, Sij_smem[br*Bc + lane_tile]);
          
          
          

          float mask_maxval;
          for (int mask=warpSize/2; mask>0; mask>>=1)
          {
            __syncwarp();
            mask_maxval = __shfl_down_sync(0xFFFFFFFF, maxval, mask);

            if (mask_maxval > maxval)
                maxval = mask_maxval;
            
          }
          maxval = __shfl_sync(0xFFFFFFFF, maxval, 0);

        }



        if(laneId==0 && br<Br && (i*Br+br)<T)
          m_smem[br] = fmaxf(last_m_smem[br], maxval);
        
        
        

        

        // Pij = exp(Sij - mi)
        if (br<Br&&(i*Br+br)<T)
        {
          for (int lane_tile=laneId; lane_tile<Bc && (j*Bc+lane_tile)<T; lane_tile+=warpSize)
            Sij_smem[br*Bc + lane_tile] = expf(Sij_smem[br*Bc + lane_tile] - m_smem[br]);
        }
        
        
        
        float sumval=0.0f;
        if (br<Br&&(i*Br+br)<T)
        {
          
          for (int lane_tile=laneId; lane_tile<Bc && (j*Bc+lane_tile)<T; lane_tile+=warpSize)
            sumval += Sij_smem[br*Bc + lane_tile];
          
          

          float mask_sumval;
          for (int mask=warpSize/2; mask>0; mask>>=1)
          {
            __syncwarp();
            mask_sumval = __shfl_down_sync(0xFFFFFFFF, sumval, mask);
            sumval += mask_sumval;
          }
          sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);
        }

        if(laneId==0 && br<Br && (i*Br+br)<T)
        {
          if(j==0)
            l_smem[br] = sumval;
          else
            l_smem[br] = expf(last_m_smem[br]-m_smem[br])*l_smem[br] + sumval;
            //l_smem[br] = expf(last_m_smem[br]-m_smem[br])*l_smem[br] + expf(maxval-m_smem[br])*sumval;
        }

      }

      __syncthreads();

      


      for (int warp_tile=0; warp_tile < std::ceil((Br*d)/warps_per_block); ++warp_tile)
      {
        int wid = warp_tile * warps_per_block + warpId;
        int br = wid / d;
        int _d = wid % d;


        float pv=0;

        if(br<Br && (i*Br+br)<T)
        {
          
          for (int lane_tile=laneId; lane_tile<Bc && (j*Bc+lane_tile)<T; lane_tile+=warpSize)
            pv += Sij_smem[br*Bc + lane_tile] * v_smem[lane_tile*d + _d];
          
          

          float mask_p;
          for (int mask=warpSize/2; mask>0; mask>>=1)
          {
            __syncwarp();
            mask_p = __shfl_down_sync(0xFFFFFFFF, pv, mask);
            pv+=mask_p;
          }
          pv = __shfl_sync(0xFFFFFFFF, pv, 0);


          
          if (laneId==0)
          { 
            if (j==0)
              o_smem[br*d + _d] = pv;
            else
              o_smem[br*d + _d] = o_smem[br*d + _d]*expf(last_m_smem[br] - m_smem[br]) + pv;
          }
        }
        
      }

      __syncthreads();




      for (int warp_tile=0; warp_tile < std::ceil(Br/warps_per_block); ++warp_tile)
      {
        int br = warp_tile*warps_per_block + warpId;
        
        if (br<Br && (i*Br+br)<T && laneId==0)
          last_m_smem[br] = m_smem[br];
        
      }
      __syncthreads();
    }
    





    // Load Oi into HBM O
    for (int warp_tile=0; warp_tile < std::ceil(Br/(float)warps_per_block); ++warp_tile)
    {
      int br = warp_tile*warps_per_block + warpId;
      
      
      
      for (int lane_tile=laneId; lane_tile<d; lane_tile+=warpSize)
      {
        if (br<Br && (i*Br+br)<T)
          o_smem[br*d + lane_tile] = o_smem[br*d + lane_tile]/l_smem[br];
      }
      
      

      __syncthreads();


      if ((i*Br+br)<T && br<Br)
      {
        for (int lane_tile=laneId; lane_tile<d; lane_tile+=warpSize)
          o[b*T*C + (i*Br+br)*C + h*d + lane_tile] = o_smem[br*d + lane_tile];
        
        if (laneId==0)
        {

          l[b*T*nh + (i*Br+br)*nh + h] = m_smem[br] + logf(l_smem[br]);
          //l[b*T*nh + (i*Br+br)*nh + h] = l_smem[br];
          //m[b*T*nh + (i*Br+br)*nh + h] = m_smem[br];
        }
      }
      
      __syncthreads();
    }
  }
}




template<int WMMA_T>
inline void blocking_wmma(const float *x, const float *w, float *o, int B, int C, int OC, cudaStream_t stream)
{
  Grid grid = CalculateBlockingSize(OC, B);
  // std::cout << "OC: " << OC << ", B: " << B << \
  //   "\ngx: " << grid.g.x << ", gy: " << grid.g.y << ", bx: " << grid.b.x << ", by: " << grid.b.y << \
  //   "\nblocking warps per block x: " << grid.wx_per_bx << ", y: " << grid.wy_per_by << \
  //   "\nx warps: " << grid.w.x/32 << ", y warps: " << grid.w.y <<  "\n\n";

  wmma_cp_async_blocking<WMMA_T, 32><<<grid.g, grid.w, grid.smem, 0>>>
                          (x, w, o, B, C, OC, grid.b.x, grid.b.y, (grid.w.x/32)*WMMA_T, grid.w.y*WMMA_T,
                          grid.wx_per_bx, grid.wy_per_by, grid.w.x/32, grid.w.y);
}


void MHSA::SetDescriptors(int B, int T, int thread_id)
{
  
  if(B!=0)
  {
    //move_to_pool();
  }

  
  //std::cout << "M: " << M << "\n";

  if (!_fp32)
  {
    Bc = 16;
    Br = 16;

    while (((2*(Br+Bc)*d) + 32*32 + 3*Br)*sizeof(float) < M && Br<32 && Bc<32)
    {
      if(Br>Bc/2)
        Br=Br*2;
      else
        Bc=Bc*2;
    }
  } else {
    
    Bc = std::ceil(  M / ((float)(4*d * sizeof(float)))  );
    Br = (int)fminf(Bc, d);
    Bc = fminf(Bc, 32);
    Br = fminf(Br, 32);    
  }


  
  
  while (((2*(Br+Bc)*d) + Br*Bc + 3*Br)*sizeof(float) > M && Br>1 && Bc>1)
  {
    if(Br>Bc/2)
      Br=(int)Br/2;
    else
      Bc=(int)Bc/2;
  }


  if (!_fp32 && (Bc<16 || Br<16))
  {
    std::string _err = "fp16 is not supported for head dimension " + std::to_string(d) + ", got Br = " + std::to_string(Br) + " and Bc = " + std::to_string(Bc) + ".\n     Falling back into floating point precision 32 (fp32).";
    LogErrorS(_err);
    _fp32 = true;
  }

    if (!_fp32 && (d%16!=0))
  {
    std::string _err = "fp16 is not supported for head dimension " + std::to_string(d) + ". It must be a multiple of 16.\n     Falling back into floating point precision 32 (fp32).";
    LogErrorS(_err);
    _fp32 = true;
  }
  

  Tc = std::ceil(T/(float)Bc);
  Tr = std::ceil(T/(float)Br);


  int last_idx = ((2*(Br+Bc)*d) + Br*Bc + 3*Br)*sizeof(float);
  std::cout << "MHSA::SetDescriptors\n - Bc: " << Bc << ";\n - Br: " << Br << ";\n - Tc: " << Tc << ";\n - Tr: " << Tr << ".\n";
  std::cout << "- M: " << M << ";\n- Last idx: " << last_idx << ".\n";
  

  this->B=B;
  this->T=T;

  qkv = get_from_pool(thread_id, B*T*3*C, "mhsa qkv");
  out = get_from_pool(thread_id, B*T*C, "mhsa out");
  l = get_from_pool(thread_id, B*T*nh, "mhsa l");

  changed_descriptors = true;
}

float *MHSA::Forward(Tensor *x, int B, int T, int thread_id)
{
  //std::cout << "MHSA::Forward" << "\n";


  

  float *proj_out = get_from_pool(thread_id, B*T*C, "mhsa out");

  cudaStream_t stream = ThreadsStream[thread_id];  

  if (this->B!=B || this->T!=T)
    SetDescriptors(B, T, thread_id);



  
  //std::cout << "" << main_stream==stream << "\n";

  if (_fp32)
  {
    // Get qkv

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(std::ceil((3*C)/(float)TILE_SIZE), std::ceil((B*T)/(float)TILE_SIZE));
    int shared_mem_size = 2*TILE_SIZE*TILE_SIZE*sizeof(float);


    const float alpha = 1.0f;
    const float beta = 0.0f;


    // std::cout << "" << shared_mem_cf << ", " << (num_warps_y*WMMA_T*WMMA_T*num_warps_x) <<  ", " << (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*2 << "\n";
    //if (thread_id==0)
      cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 3*C, B*T, C, &alpha, W, C, x->tensor_ptr, C, &beta, qkv, 3*C);
    //else
    //mult_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(x->tensor_ptr, W, qkv, TILE_SIZE, TILE_SIZE*TILE_SIZE, B*T, C, 3*C);




    
    
    //cudaStreamSynchronize(stream);
    //PrintTensorF(qkv, B*T, 3*C);

    // Attention

    dim3 grid_size_mhsa(nh, B);
    dim3 block_size_mhsa(16, WARP_SIZE);

    int threads_per_block = block_size_mhsa.x*block_size_mhsa.y;
    int warps_per_block = threads_per_block/WARP_SIZE;

    

    flash_attn_kernel<<<grid_size_mhsa, block_size_mhsa, M, stream>>>(out, qkv, l,
                                                            B, nh, T, d, C, sqrtf(d), Bc, Br, Tc, Tr, TILE_SIZE, warps_per_block, threads_per_block);
    
    //cudaStreamSynchronize(stream);
    //PrintTensorF(out, B*T, C);
    


    // Out Proj
    

    //if (thread_id==0)
      cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, C, B*T, C, &alpha, W_proj, C, out, C, &beta, proj_out, C);
    //else
    // dim3 grid_size_proj(std::ceil(C/(float)TILE_SIZE), std::ceil((B*T)/(float)TILE_SIZE));
    //mult_kernel<<<grid_size_proj, block_size, shared_mem_size, stream>>>(out, W_proj, proj_out, TILE_SIZE, TILE_SIZE*TILE_SIZE, B*T, C, C);

  } else {
    // Get qkv

    constexpr int num_warps_x{4};
    constexpr int num_warps_y{4};
    

    constexpr int WMMA_T{16};
    dim3 block_size_wmma(num_warps_x * WARP_SIZE, num_warps_y);
    dim3 grid_size_wmma_proj(std::floor((3*C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::floor((B*T + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    // int shared_mem_wmma = (num_warps_y*WMMA_T*WMMA_T*num_warps_x+ WMMA_T*WMMA_T)*sizeof(float) + (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);
    // wmma_mult_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_wmma_proj, block_size_wmma, shared_mem_wmma, stream>>>(x->tensor_ptr, W, qkv, B*T, C, 3*C);

    int shared_mem_cf = (num_warps_y*WMMA_T*WMMA_T*num_warps_x)*sizeof(float);
    // wmma_cp_async<WMMA_T,num_warps_x,num_warps_y><<<grid_size_wmma_proj, block_size_wmma, shared_mem_cf, stream>>>(x->tensor_ptr, W, qkv, B*T, C, 3*C);



    blocking_wmma<WMMA_T>(x->tensor_ptr, W, qkv, B*T, C, 3*C, stream);
  

    cudaCheck(cudaGetLastError());
    
    
    //cudaStreamSynchronize(stream);
    //PrintTensorF(qkv, B*T, 3*C);

    // Attention

    constexpr int num_warps{16};

    dim3 grid_size_mhsa(nh, B);
    dim3 block_size_mhsa(num_warps, WARP_SIZE);

    int threads_per_block = block_size_mhsa.x*block_size_mhsa.y;
    int warps_per_block = threads_per_block/WARP_SIZE;

    
    
    // std::cout << "\n\nB: " << B << ", T: " << T << ", nh: " << nh << ", d: " << d << ", C: " << C << "\n";
    // std::cout << "Launching flash attention with Bc: " << Bc << ", Br: " << Br << ", Tc " << Tc << ", Tr: " << Tr << "\n";
    // std::cout << "TILE_SIZE " << TILE_SIZE  << "\n";
    // int res = ((2*(Br+Bc)*d) + 32*32 + 3*Br);
    // std::cout << "last idx: " << res*sizeof(float) << ", M: " << M << ", warps_per_block: " << warps_per_block <<  "\n\n\n";
    


    flash_attn_kernel<<<grid_size_mhsa, block_size_mhsa, M, stream>>>(out, qkv, l,
                                                           B, nh, T, d, C, sqrtf(d), Bc, Br, Tc, Tr, TILE_SIZE, warps_per_block, threads_per_block);

    // flash_attn_fp16_kernel<WMMA_T, num_warps><<<grid_size_mhsa, block_size_mhsa, M, stream>>>(out, qkv, l,
    //                                                         B, nh, T, d, C, sqrtf(d), Bc, Br, Tc, Tr, TILE_SIZE, warps_per_block, threads_per_block);
    
    //cudaStreamSynchronize(stream);
    //PrintTensorF(out, B*T, C);
    

    // Out Proj

    // dim3 grid_size_wmma(std::floor((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::floor((B*T + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    // wmma_cp_async<WMMA_T,num_warps_x,num_warps_y><<<grid_size_wmma, block_size_wmma, shared_mem_cf, stream>>>(out, W_proj, proj_out, B*T, C, C);

    
    blocking_wmma<WMMA_T>(out, W_proj, proj_out, B*T, C, C, stream);

    cudaCheck(cudaGetLastError());
    
  }
  



  //add_forward<<<std::ceil((B*T*C)/THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(proj_out, x->tensor_ptr, proj_out, B*T*C);




  if (thread_id==0 && nn_mode==training_mode)
  {
    qkv_back = qkv;
    out_back = out;
    B_back = B;
    T_back = T;
    l_back = l;
  } else {
    move_to_pool(thread_id, B*T*3*C, qkv, "MHSA qkv");
    move_to_pool(thread_id, B*T*C, out, "MHSA pre out-proj");
    move_to_pool(thread_id, B*T*nh, l, "MHSA l");
  }
  
  return proj_out;
}





__global__ void flash_attn_backward_kernel(float *d_qkv, const float *d_o, const float *qkv, const float *o, const float *l, float *D,
                                           const int B, const int nh, const int T, const int d, const int C, const float d_scale,
                                           const int Bc, const int Br, const int Tc, const int Tr,
                                           const int warps_per_block, const int threads_per_block)
{
  int b = blockIdx.y; // batch idx
  int h = blockIdx.x; // head  idx

  if(b>=B||h>=nh)
    return;


  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int tid = (ty * blockDim.x + tx);
  
  int warpId = tid / warpSize;
  int laneId = tid % warpSize;

  
  extern __shared__ float smem[];

  float *q_smem     = smem;
  float *k_smem     = smem + Br*d;
  float *v_smem     = smem + (Br+Bc)*d;
  float *o_smem     = smem + (Br+2*Bc)*d;

  float *d_q_smem   = smem + (2*Br+2*Bc)*d;
  float *d_k_smem   = smem + (3*Br+2*Bc)*d;
  float *d_v_smem   = smem + (3*Br+3*Bc)*d;
  float *d_o_smem   = smem + (3*Br+4*Bc)*d;

  float *Sij_smem   = smem + (4*Br+4*Bc)*d;
  float *d_Pij_smem = smem + (4*Br+4*Bc)*d + Br*Bc;
  float *d_Sij_smem = smem + (4*Br+4*Bc)*d + 2*Br*Bc;
  float *l_smem     = smem + (4*Br+4*Bc)*d + 3*Br*Bc;
  float *D_smem     = smem + (4*Br+4*Bc)*d + 3*Br*Bc + Br;




  for (int tile=0; tile<ceilf((T*d)/(float)threads_per_block); ++tile)
  {
    int tile_idx = tile*threads_per_block + tid;
    int  t = tile_idx / d;
    int _d = tile_idx % d;

    if (t<T)
      d_qkv[b*T*3*C + t*3*C + h*d + _d] = 0.0f;
  }


  for (int warp_tile=0; warp_tile<std::ceil(T/warps_per_block); ++warp_tile)
  {
    int t = warp_tile * warps_per_block + warpId;

    if (t<T)
    {
      __syncwarp();
      float sumval=0.0f;
      for (int lane_tile=laneId; lane_tile<d; lane_tile+=warpSize)
        sumval += d_o[b*T*C + t*C + h*d + lane_tile]*o[b*T*C + t*C + h*d + lane_tile];

      float mask_sumval;
      for (int mask = warpSize/2; mask>0; mask>>=1)
      {
        __syncwarp();
        mask_sumval = __shfl_down_sync(0xFFFFFFFF, sumval, mask);
        sumval += mask_sumval;
      }
      sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);

      D[b*T*nh + t*nh + h] = sumval;
    }
  }


  __syncthreads();

  for(int j=0; j<Tc; ++j)
  {

    // Load K, V, init dK, dV (0)

    for (int tile=0; tile<std::ceil((Bc*d)/(float)threads_per_block); ++tile)
    {
      int tile_idx = tile*threads_per_block + tid;
      int bc = tile_idx / d;
      int _d = tile_idx % d;

      if (bc<Bc)
      {
        d_k_smem[bc*d + _d] = 0.0f;
        d_v_smem[bc*d + _d] = 0.0f;

        if ((j*Bc+bc)<T)
        {
          k_smem[bc*d + _d] = qkv[b*T*3*C + (j*Bc+bc)*3*C +   C + h*d + _d];
          v_smem[bc*d + _d] = qkv[b*T*3*C + (j*Bc+bc)*3*C + 2*C + h*d + _d];
        } else {
          k_smem[bc*d + _d] = 0.0f;
          v_smem[bc*d + _d] = 0.0f;
        }
      }
    }


    for(int i=0; i<Tr; ++i)
    {

      for (int tile=0; tile<std::ceil((Br*d)/(float)threads_per_block); ++tile)
      {
        int tile_idx = tile*threads_per_block + tid;
        int br = tile_idx / d;
        int _d = tile_idx % d;

        if(br<Br)
        {
          if((i*Br+br)<T)
          {
            if(_d==0)
            {
              D_smem[br] = D[b*T*nh + (i*Br+br)*nh + h];
              l_smem[br] = l[b*T*nh + (i*Br+br)*nh + h];
            }
            q_smem[br*d + _d]   =   qkv[b*T*3*C + (i*Br+br)*3*C + h*d + _d];
            d_q_smem[br*d + _d] = d_qkv[b*T*3*C + (i*Br+br)*3*C + h*d + _d];
            o_smem[br*d + _d]   =   o[b*T*C + (i*Br+br)*C + h*d + _d];
            d_o_smem[br*d + _d] = d_o[b*T*C + (i*Br+br)*C + h*d + _d];
          } else {
            if(_d==0)
            {
              D_smem[br] = 0.0f;
              l_smem[br] = 0.0f;
            }
            q_smem[br*d + _d]   = 0.0f;
            d_q_smem[br*d + _d] = 0.0f;
            o_smem[br*d + _d]   = 0.0f;
            d_o_smem[br*d + _d] = 0.0f;
          }
        }
      }

      __syncthreads();



      // Compute probs
      for (int warp_tile=0; warp_tile < std::ceil((Br*Bc)/(float)warps_per_block); ++warp_tile)
      {
        int wid = warp_tile * warps_per_block + warpId;
        int br = wid / Bc;
        int bc = wid % Bc;

        if (br<Br)
          Sij_smem[br*Bc + bc] = 0.0f;

        if (br<Br && (i*Br+br)<T && (j*Bc+bc)<T)
        {
          __syncwarp();
          float sij=0.0f;
          // \sum_i q[i] @ k[i].T  for each lane
          for (int lane_tile=laneId; lane_tile<d; lane_tile+=warpSize)
            sij += q_smem[br*d + lane_tile]*k_smem[bc*d + lane_tile];
          

          // \sum_i q[i] @ k[i].T  across the warp
          float mask_sij;
          for (int mask = warpSize/2; mask>0; mask>>=1)
          {
            __syncwarp();
            mask_sij = __shfl_down_sync(0xFFFFFFFF, sij, mask);
            sij += mask_sij;
          }
          sij = __shfl_sync(0xFFFFFFFF, sij, 0);

          sij = sij/d_scale;

          if (laneId==0)
            Sij_smem[br*Bc + bc] = expf(sij-l_smem[br]);
        }
      }

      __syncthreads();

      // dV
      for (int warp_tile=0; warp_tile < std::ceil((Bc*d)/(float)warps_per_block); ++warp_tile)
      {
        int wid = warp_tile * warps_per_block + warpId;
        int bc = wid / d;
        int _d = wid % d;

        if (bc<Bc && (j*Bc+bc)<T)
        {
          __syncwarp();
          float sumval=0.0f;
          // \sum_i q[i] @ k[i].T  for each lane
          for (int lane_tile=laneId; lane_tile<Br && (i*Br+lane_tile)<T; lane_tile+=warpSize)
            sumval += Sij_smem[lane_tile*Bc + bc] * d_o_smem[lane_tile*d+_d];

          // \sum_i q[i] @ k[i].T  across the warp
          float mask_sumval;
          for (int mask = warpSize/2; mask>0; mask>>=1)
          {
            __syncwarp();
            mask_sumval = __shfl_down_sync(0xFFFFFFFF, sumval, mask);
            sumval += mask_sumval;
          }
          sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);

          

          if(laneId==0)
            d_v_smem[bc*d + _d] += sumval;
        }
      }


      // d_Pij
      for (int warp_tile=0; warp_tile < std::ceil((Br*Bc)/(float)warps_per_block); ++warp_tile)
      {
        int wid = warp_tile * warps_per_block + warpId;
        int br = wid / Bc;
        int bc = wid % Bc;

        if(br<Br)
          d_Pij_smem[br*Bc + bc]=0.0f;

        if (br<Br && (i*Br+br)<T && (j*Bc+bc)<T)
        {
          __syncwarp();
          float sumval=0.0f;
          for (int lane_tile=laneId; lane_tile<d; lane_tile+=warpSize)
            sumval += d_o_smem[br*d + lane_tile] * v_smem[bc*d + lane_tile];
          

          float mask_sumval;
          for (int mask = warpSize/2; mask>0; mask>>=1)
          {
            __syncwarp();
            mask_sumval = __shfl_down_sync(0xFFFFFFFF, sumval, mask);
            sumval += mask_sumval;
          }
          sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);

          if(laneId==0)
            d_Pij_smem[br*Bc + bc] = sumval;
        }
      }

      __syncthreads();

      // d_Sij
      for (int tile=0; tile<std::ceil((Br*Bc)/(float)threads_per_block); ++tile)
      {
        int tile_idx = tile*threads_per_block + tid;
        int br = tile_idx / Bc;
        int bc = tile_idx % Bc;

        if (br<Br)
        {
          if ((i*Br+br)<T && (j*Bc+bc)<T)
            d_Sij_smem[br*Bc + bc] = Sij_smem[br*Bc + bc] * (d_Pij_smem[br*Bc + bc] - D_smem[br]);
          else
            d_Sij_smem[br*Bc + bc] = 0.0f;
        }
      }

      __syncthreads();


      // dQ
      for (int warp_tile=0; warp_tile < std::ceil((Br*d)/(float)warps_per_block); ++warp_tile)
      {
        int wid = warp_tile * warps_per_block + warpId;
        int br = wid / d;
        int _d = wid % d;

        if (br<Br && (i*Br+br)<T)
        {
          __syncwarp();
          float sumval=0.0f;

          for (int lane_tile=laneId; lane_tile<Bc && (j*Bc+lane_tile)<T; lane_tile+=warpSize)
            sumval += d_Sij_smem[br*Bc + lane_tile] * k_smem[lane_tile*d + _d];
            

          float mask_sumval;
          for (int mask = warpSize/2; mask>0; mask>>=1)
          {
            __syncwarp();
            mask_sumval = __shfl_down_sync(0xFFFFFFFF, sumval, mask);
            sumval += mask_sumval;
          }
          sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);

          if(laneId==0 && (i*Br+br)<T)
          {
            d_q_smem[br*d + _d] += sumval;
            d_qkv[b*T*3*C + (i*Br+br)*3*C + h*d + _d] = d_q_smem[br*d + _d];
          }
        }
      }


      // dK
      for (int warp_tile=0; warp_tile < std::ceil((Bc*d)/(float)warps_per_block); ++warp_tile)
      {
        int wid = warp_tile * warps_per_block + warpId;
        int bc = wid / d;
        int _d = wid % d;

        if (bc<Bc && (j*Bc+bc)<T)
        {
          __syncwarp();
          float sumval=0.0f;

          for (int lane_tile=laneId; lane_tile<Br && (i*Br+lane_tile)<T; lane_tile+=warpSize)
            sumval += d_Sij_smem[lane_tile*Bc + bc] * q_smem[lane_tile*d + _d];
            

          float mask_sumval;
          for (int mask = warpSize/2; mask>0; mask>>=1)
          {
            __syncwarp();
            mask_sumval = __shfl_down_sync(0xFFFFFFFF, sumval, mask);
            sumval += mask_sumval;
          }
          sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);

          if(laneId==0)
            d_k_smem[bc*d + _d] += sumval;
        }
      }
      __syncthreads();
    }

    // Store dK, dV to HBM

    for (int tile=0; tile<std::ceil((Bc*d)/(float)threads_per_block); ++tile)
    {
      int tile_idx = tile*threads_per_block + tid;
      int bc = tile_idx / d;
      int _d = tile_idx % d;

      if (bc<Bc && (j*Bc+bc)<T)
      {
        d_qkv[b*T*3*C + (j*Bc+bc)*3*C +   C + h*d + _d] = d_k_smem[bc*d + _d];
        d_qkv[b*T*3*C + (j*Bc+bc)*3*C + 2*C + h*d + _d] = d_v_smem[bc*d + _d];
      }
    }


    __syncthreads();
  }
}






void MHSA::SetBackwardDescriptors()
{


  _fp32_back=true;

  if (!_fp32_back)
  {
    Bc_back = 16;
    Br_back = 16;

    while (((2*(Br_back+Bc_back)*d) + Br_back*Bc_back + 3*Br_back)*sizeof(float) < M && Br_back<32 && Bc_back<32)
    {
      if(Br_back>Bc_back/2)
        Br_back=Br_back*2;
      else
        Bc_back=Bc_back*2;
    }
  } else {
    Bc_back = std::ceil(  M / ((float)(4*d * sizeof(float)))  );
    Br_back = (int)fminf(Bc_back, d);
    Bc_back = fminf(Bc_back, 32);
    Br_back = fminf(Br_back, 32);
  }
    

  
  

  while (((4*Bc_back + 4*Br_back)*d + 3*Br_back*Bc_back + 2*Br_back)*sizeof(float) > M && Br_back>1 && Bc_back>1)
  {
    if(Br_back>Bc_back/2)
      Br_back=(int)Br_back/2;
    else
      Bc_back=(int)Bc_back/2;
  }
  

  if (!_fp32_back && (Bc_back<16 || Br_back<16))
  {
    std::string _err = "fp16 is not supported for head dimension " + std::to_string(d) + ", got Br = " + std::to_string(Br_back) + " and Bc = " + std::to_string(Bc_back) + ".\n     Falling back into floating point precision 32 (fp32) at the backward mode.";
    LogErrorS(_err);
    _fp32_back = true;
  }

  Tc_back = std::ceil(T_back/(float)Bc_back);
  Tr_back = std::ceil(T_back/(float)Br_back);


  int last_idx = ((4*Bc_back + 4*Br_back)*d + 3*Br_back*Bc_back + 2*Br_back)*sizeof(float);
  std::cout << "MHSA::SetBackwardDescriptors\n - Bc: " << Bc_back << ";\n - Br: " << Br_back << ";\n - Tc: " << Tc_back << ";\n - Tr: " << Tr_back << ".\n";
  std::cout << "- M: " << M << ";\n- Last idx: " << last_idx << ".\n";

  changed_descriptors=false;
}


void MHSA::FirstBackward()
{
  dW = get_from_pool(0, 3*C*C, "MHSA dW");
  dW_proj = get_from_pool(0, C*C, "MHSA dW");

  set_to_zero_kernel<<<std::ceil((3*C*C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dW, 3*C*C);
  set_to_zero_kernel<<<std::ceil((C*C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dW_proj, C*C);

  NamedParamGrads[Name+"W"] = dW;
  NamedParamGrads[Name+"W_proj"] = dW_proj;

  first_backward=false;
}



void MHSA::Backward(float *x, float *dx, float *dy)
{
  if (changed_descriptors)
    SetBackwardDescriptors();

  if (first_backward)
    FirstBackward();

  float *d_out = get_from_pool(0, B_back*T_back*C, "MHSA d_attn");
  float *d_qkv = get_from_pool(0, B_back*T_back*3*C, "MHSA d_qkv");
  float *D = get_from_pool(0, B_back*T_back*nh, "MHSA backward D");
  float *D_aux = get_from_pool(0, B_back*T_back*T_back*nh, "MHSA backward D");


  float one = 1.0f, zero = 0.0f;
  

  if (_fp32_back)
  {
    
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size_dwproj(std::ceil(C/(float)TILE_SIZE), std::ceil(C/(float)TILE_SIZE));
    dim3 grid_size_dxproj(std::ceil(C/(float)TILE_SIZE), std::ceil((B*T)/(float)TILE_SIZE));
    int shared_mem_size = 2*TILE_SIZE_SQ*sizeof(float);


    //cudaStream_t dw_proj_stream, dw_stream;
    //cudaStreamCreate(&dw_proj_stream);
    //cudaStreamCreate(&dw_stream);


    
    //StreamAwaitStreamB(dw_proj_stream, main_stream);
    


    //mult_backwarddw<<<grid_size_dwproj, block_size, shared_mem_size, main_stream>>>(out, dW_proj, dy, TILE_SIZE, TILE_SIZE_SQ, B*T, C, C);
    //mult_backwarddx<<<grid_size_dwproj, block_size, shared_mem_size, main_stream>>>(W_proj, d_out, dy, TILE_SIZE, TILE_SIZE_SQ, B*T, C, C);


    
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, C, B*T, &one,
                              out, CUBLAS_LOWP, C, dy, CUBLAS_LOWP, C, &one,
                              dW_proj, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, C, &one,
                              W_proj, CUBLAS_LOWP, C, dy, CUBLAS_LOWP, C, &zero,
                              d_out, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    



    //PrintTensorF(d_out, T, C);


    dim3 grid_size_mhsa(nh, B);
    dim3 block_size_mhsa(16, WARP_SIZE);

    int threads_per_block = block_size_mhsa.x*block_size_mhsa.y;
    int warps_per_block = threads_per_block/WARP_SIZE;


    //int last_id = ((4*Bc_back + 4*Br_back)*d + 3*Br_back*Bc_back + 2*Br_back)*sizeof(float);
    //std::cout << "backward last_id: " << last_id << ", M: " << M << "\n";
    //std::cout << "Bc: " << Bc_back << ", Br: " << Br_back << ", Tc: " << Tc_back << ", Tr: " << Tr_back << "\n";
    flash_attn_backward_kernel<<<grid_size_mhsa, block_size_mhsa, M, main_stream>>>(d_qkv, d_out, qkv, out, l, D,
                                                                                B, nh, T, d, C, sqrtf(d),
                                                                                Bc_back, Br_back, Tc_back, Tr_back,
                                                                                warps_per_block, threads_per_block);

    //PrintTensorF(d_qkv, 3, C);
    //PrintTensorF(d_qkv, T, 3*C);
    
    //StreamAwaitStreamB(dw_stream, main_stream);

    dim3 grid_size_dx(std::ceil(C/(float)TILE_SIZE), std::ceil((B*T)/(float)TILE_SIZE));
    dim3 grid_size_dw(std::ceil(C/(float)TILE_SIZE), std::ceil((3*C)/(float)TILE_SIZE));
    //mult_backwarddw<<<grid_size_dw, block_size, shared_mem_size, main_stream>>>(x, dW, d_qkv, TILE_SIZE, TILE_SIZE_SQ, B*T, C, 3*C);
    //mult_backwarddx<<<grid_size_dx, block_size, shared_mem_size, main_stream>>>(W, dx, d_qkv, TILE_SIZE, TILE_SIZE_SQ, B*T, C, 3*C);
    
    
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, 3*C, B*T, &one,
                              x, CUBLAS_LOWP, C, d_qkv, CUBLAS_LOWP, 3*C, &one,
                              dW, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, 3*C, &one,
                              W, CUBLAS_LOWP, C, d_qkv, CUBLAS_LOWP, 3*C, &zero,
                              dx, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    

    //StreamAwaitStreamB(main_stream, dw_proj_stream);
    //StreamAwaitStreamB(main_stream, dw_stream);
    //cudaStreamDestroy(dw_proj_stream);
    //cudaStreamDestroy(dw_stream);
  } else {
    

    //cudaStream_t dw_proj_stream, dw_stream;
    //cudaStreamCreate(&dw_proj_stream);
    //cudaStreamCreate(&dw_stream);


    
    //StreamAwaitStreamB(dw_proj_stream, main_stream);
    
    constexpr int num_warps_x{4};
    constexpr int num_warps_y{4};
    
    constexpr int WMMA_T{16};
    
    int shared_mem_size = num_warps_y*WMMA_T*WMMA_T*num_warps_x*sizeof(float) + (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);

    dim3 block_size(num_warps_x * WARP_SIZE, num_warps_y);

    dim3 grid_size_dx_proj(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((B*T + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    dim3 grid_size_dw_proj(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((C + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));

    wmma_backwarddx_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dx_proj, block_size, shared_mem_size, main_stream>>>(d_out, W_proj, dy, B*T, C, C);
    wmma_backwarddw_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dw_proj, block_size, shared_mem_size, main_stream>>>(dW_proj, out, dy, B*T, C, C);
    
    


    dim3 grid_size_mhsa(nh, B);
    dim3 block_size_mhsa(16, WARP_SIZE);

    int threads_per_block = block_size_mhsa.x*block_size_mhsa.y;
    int warps_per_block = threads_per_block/WARP_SIZE;


    //int last_id = ((4*Bc_back + 4*Br_back)*d + 3*Br_back*Bc_back + 2*Br_back)*sizeof(float);
    //std::cout << "backward last_id: " << last_id << ", M: " << M << "\n";
    //std::cout << "Bc: " << Bc_back << ", Br: " << Br_back << ", Tc: " << Tc_back << ", Tr: " << Tr_back << "\n";
    flash_attn_backward_kernel<<<grid_size_mhsa, block_size_mhsa, M, main_stream>>>(d_qkv, d_out, qkv, out, l, D,
                                                                                B, nh, T, d, C, sqrtf(d),
                                                                                Bc_back, Br_back, Tc_back, Tr_back,
                                                                                warps_per_block, threads_per_block);

    //PrintTensorF(d_qkv, 3, C);
    //PrintTensorF(d_qkv, T, 3*C);
    
    //StreamAwaitStreamB(dw_stream, main_stream);



    dim3 grid_size_dx(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((B*T + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    dim3 grid_size_dw(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((3*C + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));

    wmma_backwarddx_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dx, block_size, shared_mem_size, main_stream>>>(dx, W, d_qkv, B*T, C, 3*C);
    wmma_backwarddw_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dw, block_size, shared_mem_size, main_stream>>>(dW, x, d_qkv, B*T, C, 3*C);
    
    
  

    //StreamAwaitStreamB(main_stream, dw_proj_stream);
    //StreamAwaitStreamB(main_stream, dw_stream);
    //cudaStreamDestroy(dw_proj_stream);
    //cudaStreamDestroy(dw_stream);

  }







  //add_forward<<<std::ceil((B*T*C)/THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, main_stream>>>(dx, dx, dy, B*T*C);


  // Clean-up
  move_to_pool(0, B_back*T_back*C, d_out, "MHSA d_attn");
  move_to_pool(0, B_back*T_back*3*C, d_qkv, "MHSA d_qkv");
  move_to_pool(0, B_back*T_back*nh, D, "MHSA backward D");
  move_to_pool(0, B_back*T_back*T_back*nh, D_aux, "MHSA backward D");

}


void mhsa_backward(float *x, float *dx, float *dy, std::string name)
{
  std::unique_ptr<MHSA> mhsa = std::move(NamedMHSA[name]);

  mhsa->Backward(x, dx, dy);

  NamedMHSA[name] = std::move(mhsa);
}


extern "C" void *MHSAForward(char *self, Tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self)
{
  //TODO: remove self arg and concatenate it instead during the function call
  
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  //std::cout << "\nMHSA forward of " << conv_name << " with input " << tensor_q->name  << "\n";
  //std::cout << "thread id: " << thread_id << "\n\n";

  

  float *output;
  
  std::vector<float> dims = tensor->dims;
  

  float B = dims[0];
  float T = dims[dims.size()-2];
  float C = dims[dims.size()-1];

  //std::vector<float> new_dims = dims;
  std::vector<float> new_dims = {B, T, C};




  std::unique_ptr<MHSA> mhsa = std::move(NamedMHSA[conv_name]);

  if ((int)C!=(int)mhsa->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string((int)C) + ", while the expected input channels of the MHSA are: " + std::to_string(mhsa->C);
    LogError(error);
    
    NamedMHSA[conv_name] = std::move(mhsa);
    return nullptr;
  }



  tensor->Sync();
  output = mhsa->Forward(tensor, (int) B, (int)T, thread_id);





  NamedMHSA[conv_name] = std::move(mhsa);

  //std::cout << "Return with dims:" << "\n";
  //PrintDims(new_dims);
  

  Tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, "");
  new_tensor->AttrLNode(tensor, mhsa_op);
  new_tensor->scopeless_name = conv_name;
  return new_tensor;
}















void Linear::SetDescriptors(int B, int thread_id)
{
  this->B=B;
  changed_descriptors=true;
}













float *Linear::Forward(Tensor *x, int thread_id)
{

  std::vector<float> dims = format_LinearLayer_Dims(x->dims);
  int B = dims[0];

  if (this->B!=B)
    SetDescriptors(B, thread_id);


  float *out = get_from_pool(thread_id, B*OC, "linear fwd");

  cudaStream_t stream = ThreadsStream[thread_id];  


  if (_fp32)
  {
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(std::ceil(OC/(float)TILE_SIZE), std::ceil(B/(float)TILE_SIZE));
    int shared_mem_size = 2*TILE_SIZE*TILE_SIZE*sizeof(float);

    mult_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(x->tensor_ptr, W, out, TILE_SIZE, TILE_SIZE*TILE_SIZE, B, C, OC);
  } else {
    
    constexpr int num_warps_x{4};
    constexpr int num_warps_y{4};
    

    constexpr int WMMA_T{16};
    dim3 block_size(num_warps_x * WARP_SIZE, num_warps_y);
    dim3 block_size_pp(num_warps_x * WARP_SIZE*2, num_warps_y);
    dim3 grid_size(std::floor((OC + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::floor((B + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));

    // dim3 grid_size_pp(std::ceil((OC + ((num_warps_x/2)*WMMA_T - 1)) / (float)((num_warps_x/2)*WMMA_T)), std::ceil((B + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));

    
    // int shared_mem_pp   = (num_warps_y*WMMA_T*WMMA_T*(num_warps_x/2))*sizeof(float) + 2*((num_warps_x/2)+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);
    int shared_mem_pp   = (num_warps_y*WMMA_T*WMMA_T*num_warps_x)*sizeof(float) + 2*(num_warps_x+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);





    // int shared_mem_cf = (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*2*sizeof(float);
    // wmma_cp_async<WMMA_T,num_warps_x,num_warps_y><<<grid_size, block_size, shared_mem_cf, stream>>>(x->tensor_ptr, W, out, B, C, OC);





    blocking_wmma(x->tensor_ptr, W, out, B, C, OC, stram);
    
    



    // float *bank;
    // cudaMalloc(&bank, 16*32*4);
    
    // std::cout << "\n\n\n";
    // std::cout << "B: " << B << ", C: " << C << ", OC: " << OC << "\nbx " << grid_size.x << ", by " << grid_size.y << "\n\n";

    // int shared_mem_size = (num_warps_y*WMMA_T*WMMA_T*num_warps_x)*sizeof(float) +   (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);
    // wmma_mult_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size, block_size, shared_mem_size, stream>>>(x->tensor_ptr, W, out, B, C, OC);
    // wmma_pingpong<WMMA_T,num_warps_x,num_warps_y><<<grid_size, block_size_pp, shared_mem_pp, stream>>>(x->tensor_ptr, W, out, B, C, OC);

    // PrintTensorF(bank, 32, 16);

    // wmma_mult_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size, block_size, shared_mem_size, stream>>>(x->tensor_ptr, W, out, B, C, OC);


    // PrintTensorF(out, 16, 16);

    cudaCheck(cudaGetLastError());

  }
  
  return out;
}



void Linear::SetBackwardDescriptors()
{
  changed_descriptors=false;
}

void Linear::FirstBackward()
{
  dW = get_from_pool(0, OC*C, "MHSA dW");

  set_to_zero_kernel<<<std::ceil((OC*C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dW, OC*C);

  NamedParamGrads[Name+"W"] = dW;

  first_backward=false;
}


void Linear::Backward(float *x, float *dx, float *dy)
{
  float one = 1.0f, zero = 0.0f;
  
  
  if(first_backward)
    FirstBackward();
  //if(changed_descriptors)
  //  SetBackwardDescriptors();


  if (_fp32)
  {
  // backwad to dx
  cublasCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B, OC, &one,
                             W, CUBLAS_LOWP, C, dy, CUBLAS_LOWP, OC, &zero,
                             dx, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  
  
  // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
  cublasCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B, &one,
                             x, CUBLAS_LOWP, C, dy, CUBLAS_LOWP, OC, &one,
                             dW, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {

    constexpr int num_warps_x{4};
    constexpr int num_warps_y{4};
    

    constexpr int WMMA_T{16};
    dim3 block_size(num_warps_x * WARP_SIZE, num_warps_y);
    dim3 grid_size_dx(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((B + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    dim3 grid_size_dw(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((OC + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    
    int shared_mem_size = num_warps_y*WMMA_T*WMMA_T*num_warps_x*sizeof(float) + (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);

    int shared_mem_cf = (num_warps_y*WMMA_T*WMMA_T*num_warps_x)*sizeof(float) +   (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*2*sizeof(float);

    // wmma_dx_cp_async<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dx, block_size, shared_mem_cf, main_stream>>>(dx, W, dy, B, C, OC);
    wmma_backwarddx_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dx, block_size, shared_mem_size, main_stream>>>(dx, W, dy, B, C, OC);
    wmma_backwarddw_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dw, block_size, shared_mem_size, main_stream>>>(dW, x, dy, B, C, OC);




    /*
    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    cublasCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B, &one,
                             x, CUBLAS_LOWP, C, dy, CUBLAS_LOWP, OC, &one,
                             dW, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    */
  }
}

void linear_backward(float *x, float *dx, float *dy, std::string name)
{
  std::unique_ptr<Linear> linear = std::move(NamedLinear[name]);

  linear->Backward(x, dx, dy);

  NamedLinear[name] = std::move(linear);
}





extern "C" void *LinearForward(char *self, Tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self)
{
  //TODO: remove self arg and concatenate it instead during the function call
  
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  //std::cout << "\nLinear of " << conv_name << " with input " << tensor->name  << "\n";
  //std::cout << "thread id: " << thread_id << "\n\n";

  

  float *output;
  
  std::vector<float> dims = tensor->dims;
  
  int C = dims[dims.size()-1];



  std::unique_ptr<Linear> linear = std::move(NamedLinear[conv_name]);

  if ((int)C!=(int)linear->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string((int)C) + ", while the expected input channels of the Linear are: " + std::to_string(linear->C);
    LogError(error);
    
    NamedLinear[conv_name] = std::move(linear);
    return nullptr;
  }
    
  std::vector<float> new_dims = RemoveLastDim(dims);
  new_dims.push_back(linear->OC);
  



  tensor->Sync();
  output = linear->Forward(tensor, thread_id);


  NamedLinear[conv_name] = std::move(linear);

  

  Tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, "");
  new_tensor->AttrLNode(tensor, linear_op);
  new_tensor->scopeless_name = conv_name;
  return new_tensor;
}

















__global__ void lstm_single_step_kernel(float *fused_out, const float *x_out, const float *W, const float *ht, const float *b,
                      const int t, const int T, const int tile_size, const int tile_offset,
                      const int B, const int OC, const int fourX_OC, const int tanh_offset) {
  // x_out  e [B,  4*OC]
  // ht     e [T, B, OC]
  // W      e [4*OC, OC]

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x_block = blockIdx.x;
  int y_block = blockIdx.y;

  

  int row = y_block*tile_size + ty;
  int col = x_block*tile_size + tx;



  int offset = tile_offset;

  float y = 0.0f;


  extern __shared__ float smem[];


  

  if (t>0)
  {
    for (int i=0; i < ceilf(OC/(float)tile_size); ++i)
    {
      // each tile has a subset of columns to work with
      // tile_tid tells which exact column to use from the subset
      // it assumes that W is transposed already

      int _col  = i * tile_size + tx;
      int _col2 = i * tile_size + ty;
      
      if(row<B && _col<OC)
        smem[tx* tile_size +ty] = ht[((t-1)*B + row)*OC + _col];
      else
        smem[tx* tile_size +ty] = 0;
      
      if (col<fourX_OC && _col2<OC)
        smem[offset+ty* tile_size +tx] = W[col*OC + _col2];
      else
        smem[offset+ty* tile_size +tx] = 0;
      
      __syncthreads();


      for(int j=0; j<tile_size; ++j)
        y += smem[j* tile_size +ty] * smem[offset+j* tile_size +tx];
      
      __syncthreads();
      
    }
  }



  if(row<B && col<fourX_OC)
  {
    
    if (col<tanh_offset)
    {
      if (t==0) //TODO: maybe create a separate kernel to solve the ifs?
        y = 1/(1+exp(-(     x_out[(row*T + t)*fourX_OC + col] +b[col]) ));
      else
        y = 1/(1+exp(-( y + x_out[(row*T + t)*fourX_OC + col] +b[col]) ));
    }
    else
    {
      if (t==0)
        y = tanhf(     x_out[(row*T + t)*fourX_OC + col] +b[col]);
      else
        y = tanhf( y + x_out[(row*T + t)*fourX_OC + col] +b[col]);
    }

    // Now we have tensors i, f, o and c_
    // Output dim is: [T, B, 4*OC]
    // Continuing on this kernel will result on partial usage of this kernel threads, we therefore move the result to the global memory and call another kernel

    fused_out[(t*B + row)*fourX_OC + col] = y;
  }
}






__global__ void lstm_elementwise_ops_kernel(const float *fused_out,
                      float *ht, float *ct,
                      const int tile_size, const int tile_offset,
                      const int t, const int T,
                      const int B, const int OC, const int fourX_OC,
                      const int f_offset, const int o_offset, const int c_offset) {
  // ht        e [T, B,    OC]
  // ct        e [T, B,    OC]
  // fused out e [T, B,  4*OC]

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x_block = blockIdx.x;
  int y_block = blockIdx.y;
  

  int row = y_block*tile_size + ty;
  int col = x_block*tile_size + tx;


  if(row<B && col<OC)
  {
    float _ct;
    float idx = (t*B + row);
    int tb_offset = idx*fourX_OC; //TODO: factorize index
    int ht_tb_offset = idx*OC;

    // ct = f*ct + i*c_
    if(t==0)
      _ct = fused_out[tb_offset + col]*fused_out[tb_offset + c_offset + col];
    else
      _ct = fused_out[tb_offset + f_offset + col]*ct[((t-1)*B + row)*OC + col] + fused_out[tb_offset + col]*fused_out[tb_offset + c_offset + col];

    // ht = o*tanh(ct)
    ht[ht_tb_offset + col] = fused_out[tb_offset + o_offset + col]*tanhf(_ct);
    ct[ht_tb_offset + col] = _ct;
  }
}



void LSTM::SetDescriptors(int B, int T, int thread_id)
{

  if (x_out!=nullptr)
  {
    cudaFree(x_out);
    cudaFree(fused_out);
    cudaFree(all_ht);
    cudaFree(all_ct);
  }

  x_out = get_from_pool(thread_id, B*T*4*OC, "lstm x@U out");
  fused_out = get_from_pool(thread_id, T*B*4*OC, "lstm ht@W");

  all_ht = get_from_pool(thread_id, T*B*OC, "lstm all ht");
  all_ct = get_from_pool(thread_id, T*B*OC, "lstm all ct");


  int grid_size, block_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(T*B*OC);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];


  //set_to_zero_kernel<<<grid_size, block_size, 0, main_stream>>>(all_ht, B*T*OC);


  this->B=B;
  this->T=T;
  changed_descriptors=true;
}


float *LSTM::Forward(Tensor *tensor_x, Tensor *tensor_ht, Tensor *tensor_ct, int B, int T, int thread_id)
{

  dim3 block_size(TILE_SIZE, TILE_SIZE);
  dim3 grid_size(  std::ceil( (4*OC) / (float)TILE_SIZE),   std::ceil( (B*T) / (float)TILE_SIZE)  );
  int shared_mem_size = 2*TILE_SIZE_SQ*sizeof(float);


  
  
  if (B!=this->B || T!=this->T)
    SetDescriptors(B,T,thread_id);
  
  cudaStream_t stream = ThreadsStream[thread_id];
  //cudaStream_t stream = main_stream;





  

  //mult_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(tensor_x->tensor_ptr, U, x_out, TILE_SIZE, TILE_SIZE_SQ, B*T, C, 4*OC);


  
  constexpr int num_warps_x{8};
  constexpr int num_warps_y{4};
  

  constexpr int WMMA_T{16};
  dim3 block_size_wmma(num_warps_x * WARP_SIZE, num_warps_y);
  dim3 grid_size_wmma(std::ceil((4*OC + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((B*T + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
  int shared_mem_wmma = num_warps_y*WMMA_T*WMMA_T*num_warps_x*sizeof(float) + (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);
  wmma_mult_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_wmma, block_size_wmma, shared_mem_wmma, stream>>>(tensor_x->tensor_ptr, U, x_out, B*T, C, 4*OC);
  





  //move_to_pool(tensor_ht->dims_prod, tensor_ht->tensor_ptr, "input ht");
  //move_to_pool(tensor_ct->dims_prod, tensor_ct->tensor_ptr, "input ct");
  
  

  dim3 grid_size_lstm(  std::ceil( (4*OC) / (float)TILE_SIZE),   std::ceil( B / (float)TILE_SIZE)  );
  dim3 grid_size_elementwises(  std::ceil( (OC) / (float)TILE_SIZE),   std::ceil( B / (float)TILE_SIZE)  );



  //std::cout << "\nx out"  << "\n";
  //PrintTensorF(x_out, B*T, 4*OC);
  //std::cout << "\n";



  int f_offset =     OC;
  int o_offset = 2 * OC;
  int c_offset = 3 * OC;

  for (int t=0; t<T; ++t)
  {
    //std::cout << "Forward t: " << t << "\n";


    lstm_single_step_kernel<<<grid_size_lstm, block_size, shared_mem_size, stream>>>(fused_out, x_out, W, all_ht, b,
                                                                                      t, T, TILE_SIZE, TILE_SIZE_SQ, B, OC, 4*OC, 3*OC);

    //std::cout << "\nFused out"  << "\n";
    //PrintTensorF(fused_out, B, 4*OC);
    //std::cout << "\n";

    lstm_elementwise_ops_kernel<<<grid_size_elementwises, block_size, 0, stream>>>(fused_out,
                                                                                      all_ht, all_ct,
                                                                                      TILE_SIZE, TILE_SIZE_SQ,
                                                                                      t, T,
                                                                                      B, OC, 4*OC,
                                                                                      f_offset, o_offset, c_offset);
  }


  tensor_ht->tensor_ptr = all_ht + (int)((T-1)*B*OC);
  tensor_ct->tensor_ptr = all_ct + (int)((T-1)*B*OC);

  return tensor_ht->tensor_ptr;
}




























__global__ void lstm_single_step_backward_dht_kernel(const float *d_ifoc,
                      float *d_ht, const float *w,
                      const int t, const int _t, const int T,
                      const int tile_size, const int tile_offset,
                      const int B, const int C, const int OC) {

  int row = blockIdx.y * blockDim.y + threadIdx.y; // B
  int col = blockIdx.x * blockDim.x + threadIdx.x; // C
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // d_ht      e [B,      OC]
  // W         e [4*OC,   OC]
  // d_ifoc    e [T, B, 4*OC]



  extern __shared__ char _smem[];
  auto smem = reinterpret_cast<float*>(_smem);



  int offset = tile_offset;

  float tmp = 0.0f;
  // consider row as B and col as C
  
  __syncthreads();
  

#pragma unroll
  for (int i=0; i<ceilf(OC/(float)tile_size); ++i)
  {
    int _col = i*tile_size + tx;
    int _row = i*tile_size + ty;


    if( row<B  && _col<OC)
      smem[tx*tile_size +ty] = d_ifoc[_t*B*OC + row*OC + _col];
    else
      smem[tx*tile_size +ty] = 0;


    if(_row<OC &&  col<C)
      smem[offset+ty*tile_size +tx] = w[_row*C + col];
    else
      smem[offset+ty*tile_size +tx] = 0;
    

    __syncthreads();


#pragma unroll
    for(int j=0; j<tile_size; ++j)
      tmp += smem[j*tile_size +ty] * smem[offset+j*tile_size +tx];
    
    __syncthreads();
  }

  if(row<B && col<C)
    d_ht[row * C + col] = tmp;
}



__global__ void lstm_backward_dx_kernel(const float *d_ifoc,
                      float *dx, const float *w,
                      const int tile_size, const int tile_offset,
                      const int B, const int T, const int C, const int OC) {

  int row = blockIdx.y * blockDim.y + threadIdx.y; // BT
  int col = blockIdx.x * blockDim.x + threadIdx.x; // OC
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // d_ifoc e [T, B, 4*OC]
  // dx     e [B, T,    C]

  float sum = 0.0f;


  extern __shared__ char _smem[];
  auto smem = reinterpret_cast<float*>(_smem);



  int offset = tile_offset;

  float tmp = 0.0f;
  // consider row as BT and col as C
  
  __syncthreads();

  int b = row / T;
  int t = row % T;
  
#pragma unroll
  for (int i=0; i<ceilf(OC/(float)tile_size); ++i)
  {
    int _col = i*tile_size + tx;
    int _row = i*tile_size + ty;


    if( row<B*T  && _col<OC)
      smem[tx*tile_size +ty] = d_ifoc[(t*B + b)*OC + _col];
    else
      smem[tx*tile_size +ty] = 0;


    if(_row<OC &&  col<C)
      smem[offset+ty*tile_size +tx] = w[_row*C + col];
    else
      smem[offset+ty*tile_size +tx] = 0;
    

    __syncthreads();


#pragma unroll
    for(int j=0; j<tile_size; ++j)
      tmp += smem[j*tile_size +ty] * smem[offset+j*tile_size +tx];
    
    __syncthreads();
  }

  if(row<B*T && col<C)
    dx[(b*T + t)*C + col] = tmp;
}






__global__ void lstm_elementwise_ops_backward_kernel(const float *fused_out,
                      const float *ct,
                      float *d_ht, float *d_ct, float *d_ifoc, float *dB,
                      const float *w,
                      const int tile_size, const int tile_offset,
                      const int t, const int _t, const int T,
                      const int B, const int OC, const int fourX_OC,
                      const int f_offset, const int o_offset, const int c_offset) {
  // d_ht      e [B,      OC]
  // d_ct      e [B,      OC]
  // d_ifoc    e [T, B, 4*OC]
  // ct        e [T, B,   OC]
  // fused out e [T, B, 4*OC]

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x_block = blockIdx.x;
  int y_block = blockIdx.y;
  

  int row = y_block*tile_size + ty;
  int col = x_block*tile_size + tx;


  int tb_offset = (_t*B + row)*fourX_OC;

  if(row<B && col<OC)
  {
    float d_ct_aux, _ct, tanh_ct, _d_ht;

    _d_ht = d_ht[row*OC + col];
    _ct = ct[(_t*B + row)*OC + col];
    tanh_ct = tanhf(_ct);


    // ct = f*ct + i*c_
    // ht = o*tanh(ct)

    
    float i = fused_out[tb_offset + col];
    float f = fused_out[tb_offset + f_offset + col];
    float o = fused_out[tb_offset + o_offset + col];
    float c = fused_out[tb_offset + c_offset + col];

    float d_i, d_f, d_o, d_c;

    d_o = _d_ht * tanh_ct;

    d_ct_aux = o * _d_ht;
    d_ct_aux = (1 - tanh_ct*tanh_ct) * d_ct_aux;

    if(t!=0) // set to zero instead of accumulating on the first iter
      d_ct_aux += d_ct[row*OC + col];

    // ct = f*ct + i*c_

    d_i = c * d_ct_aux;
    d_c = i * d_ct_aux;

    d_f = _ct * d_ct_aux;
    d_ct[row*OC + col] = f * d_ct_aux;


    d_ifoc[tb_offset +            col] = (i*(1-i)) * d_i;
    d_ifoc[tb_offset + f_offset + col] = (f*(1-f)) * d_f;
    d_ifoc[tb_offset + o_offset + col] = (o*(1-o)) * d_o;
    d_ifoc[tb_offset + c_offset + col] = (1- c*c ) * d_c;


    //fused_out[_t*B*fourX_OC + row*fourX_OC + f_offset + col]
    //_ct = fused_out[_t*B*fourX_OC + row*fourX_OC + f_offset + col]*ct[_t*B*OC + row*OC + col] + fused_out[_t*B*fourX_OC + row*fourX_OC + col]*fused_out[_t*B*fourX_OC + row*fourX_OC + c_offset + col];

    /**/
    float *db = dB + col;
    atomicAdd(db,          d_ifoc[tb_offset +            col]);
    atomicAdd(db+f_offset, d_ifoc[tb_offset + f_offset + col]);
    atomicAdd(db+o_offset, d_ifoc[tb_offset + o_offset + col]);
    atomicAdd(db+c_offset, d_ifoc[tb_offset + c_offset + col]);
    
  }
  
  extern __shared__ char _smem[];
  auto smem = reinterpret_cast<float*>(_smem);

  int offset = tile_offset;

  float tmp = 0.0f;
  // consider row as B and col as C
  
  __syncthreads();
  
#pragma unroll
  for (int i=0; i<ceilf(fourX_OC/(float)tile_size); ++i)
  {
    int _col = i*tile_size + tx;
    int _row = i*tile_size + ty;


    if( row<B  && _col<fourX_OC)
      smem[tx*tile_size +ty] = d_ifoc[tb_offset + _col];
    else
      smem[tx*tile_size +ty] = 0;


    if(_row<fourX_OC &&  col<OC)
      smem[offset+ty*tile_size +tx] = w[_row*OC + col];
    else
      smem[offset+ty*tile_size +tx] = 0;
    

    __syncthreads();

#pragma unroll
    for(int j=0; j<tile_size; ++j)
      tmp += smem[j*tile_size +ty] * smem[offset+j*tile_size +tx];
    
    __syncthreads();
  }

  if(row<B && col<OC)
    d_ht[row*OC + col] = tmp;
}







void LSTM::SetBackwardDescriptors()
{
  std::cout << "Changed LSTM descriptors." << "\n";


  d_ht   = get_from_pool(0, B*OC, "d_ht");
  d_ct   = get_from_pool(0, B*OC, "d_ct");
  d_ifoc = get_from_pool(0, T*B*4*OC, "d_ct");

  changed_descriptors=false;
}

void LSTM::FirstBackward()
{
  std::cout << "First LSTM backward." << "\n";

  dW = get_from_pool(0, 4*OC*OC, "lstm dW");
  dU = get_from_pool(0, 4*OC* C, "lstm dU");
  dB = get_from_pool(0, 4*OC,    "lstm dB");

  set_to_zero_kernel<<<std::ceil((4*OC*OC)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dW, 4*OC*OC);
  set_to_zero_kernel<<<std::ceil((4*OC* C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dU, 4*OC*C);
  set_to_zero_kernel<<<std::ceil((4*OC)   /(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dB, 4*OC);

  NamedParamGrads[Name+"W"] = dW;
  NamedParamGrads[Name+"U"] = dU;
  NamedParamGrads[Name+"B"] = dB;

  first_backward=false;
}


void LSTM::Backward(float *x, float *dx, float *dy)
{
  dim3 block_size(TILE_SIZE, TILE_SIZE);
  int shared_mem_size = 2*TILE_SIZE_SQ*sizeof(float);



  if (first_backward)
    FirstBackward();
  if (changed_descriptors)
    SetBackwardDescriptors();

  

  //std::cout << "Copy dy to d_ht" << "\n";
  copy_tensor_kernel<<<std::ceil(((float)B*(float)OC)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(d_ht, dy, B*OC);
  set_to_zero_kernel<<<std::ceil(((float)B*(float)OC)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(d_ct,     B*OC); // TODO: check if removing this one is safe
  set_to_zero_kernel<<<std::ceil(((float)T*(float)B*4*(float)OC)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(d_ifoc, T*B*4*OC);


  

  //PrintTensorF(d_ht, B, OC);



  dim3 grid_size_elementwises(  std::ceil( (float)OC / (float)TILE_SIZE),   std::ceil( (float)B / (float)TILE_SIZE)  );
  dim3 grid_size_d_ht(          std::ceil( (float)OC / (float)TILE_SIZE),   std::ceil( (float)B / (float)TILE_SIZE)  );


  int f_offset =     OC;
  int o_offset = 2 * OC;
  int c_offset = 3 * OC;

  int reversed_t_;

  for (int t=0; t<T; ++t)
  {
    reversed_t_ = T-t-1;
    
    //std::cout << "backward t: " << t << ", reversed t: " << reversed_t_ << "\n";

    lstm_elementwise_ops_backward_kernel<<<grid_size_elementwises, block_size, shared_mem_size, main_stream>>>(fused_out,
                                                                                      all_ct,
                                                                                      d_ht, d_ct, d_ifoc, dB,
                                                                                      W,
                                                                                      TILE_SIZE, TILE_SIZE_SQ,
                                                                                      t, reversed_t_, T,
                                                                                      B, OC, 4*OC,
                                                                                      f_offset, o_offset, c_offset);

    //PrintTensorF(fused_out+reversed_t_*B*4*OC, B, 4*OC);
    //PrintTensorF(d_ifoc, B, OC);
    /*
    lstm_single_step_backward_dht_kernel<<<grid_size_d_ht, block_size, shared_mem_size, main_stream>>>(d_ifoc,
                                                                                      d_ht, W,
                                                                                      t, reversed_t_, T,
                                                                                      TILE_SIZE, TILE_SIZE_SQ,
                                                                                      B, OC, 4*OC);
    */
    //PrintTensorF(d_ht, B, OC);
  }
  //PrintTensorF(d_ht, B, OC);

  dim3 grid_size_dx(  std::ceil( (float)C  / (float)TILE_SIZE),   std::ceil( (float)B*(float)T   / (float)TILE_SIZE)  );
  //dim3 grid_size_dw(  std::ceil( 4*(float)OC*(float)OC / (float)TILE_SIZE_SQ)  );
  //dim3 grid_size_du(  std::ceil( 4*(float)OC*(float)C  / (float)TILE_SIZE_SQ)  );

  dim3 grid_size_dw(std::ceil(OC/(float)TILE_SIZE), std::ceil((4*OC)/(float)TILE_SIZE));
  dim3 grid_size_du(std::ceil(C /(float)TILE_SIZE), std::ceil((4*OC)/(float)TILE_SIZE));


  // all_ht    e [T, B,    OC]
  // all_ct    e [T, B,    OC]
  // fused out e [T, B,  4*OC]
  // d_ifoc    e [T, B,  4*OC]

  // x         e [B, T,    OC]



  
  cudaStream_t dx_stream, dw_stream;
  cudaStreamCreate(&dx_stream);
  cudaStreamCreate(&dw_stream);

  cudaStreamSynchronize(main_stream);

  
  lstm_backward_dx_kernel<<<grid_size_dx, block_size, shared_mem_size, dx_stream>>>(d_ifoc, dx, U,
                                                                                      TILE_SIZE, TILE_SIZE_SQ, B, T, C, 4*OC);
  RegisterEvent(dx_stream);
  
  

  
  mult_backwarddw<<<grid_size_dw, block_size, shared_mem_size, dw_stream>>>(all_ht, dW, d_ifoc, TILE_SIZE, TILE_SIZE_SQ, B*T, OC, 4*OC);
  RegisterEvent(dw_stream);

  
  mult_backwarddw<<<grid_size_du, block_size, shared_mem_size, main_stream>>>(x, dU, d_ifoc, TILE_SIZE, TILE_SIZE_SQ, B*T, C, 4*OC);

  WaitForAllEvents();
  cudaStreamDestroy(dx_stream);
  cudaStreamDestroy(dw_stream);
  

  /*
  move_to_pool(B*OC, d_ht, "d_ht");
  move_to_pool(B*OC, d_ct, "d_ct");
  move_to_pool(T*B*4*OC, d_ifoc, "d_ht");
  */
} 





void lstm_backward(float *x, float *dx, float *dy, std::string name)
{
  std::unique_ptr<LSTM> lstm = std::move(NamedLSTM[name]);

  lstm->Backward(x, dx, dy);

  NamedLSTM[name] = std::move(lstm);
}

void embedding_backward(float *x, float *dy, std::string name)
{
  std::unique_ptr<Embedding> embedding = std::move(NamedEmbedding[name]);

  embedding->Backward(x, dy);

  NamedEmbedding[name] = std::move(embedding);
}



extern "C" void *LSTMForward(char *self, Tensor *tensor_x, Tensor *tensor_ht, Tensor *tensor_ct, int thread_id, char *conv_namec, int is_obj_attr_or_self)
{
  //TODO: remove self arg and concatenate it instead during the function call
  
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  //std::cout << "LSTM forward of " << conv_name << " with input " << tensor_x->name << ", ht: " << tensor_ht->name << "\n";


  

  float *tensor_ptr, *output, *d_filter;
  
  std::vector<float> dims = tensor_x->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float T = dims[dims.size()-2];
  float C = dims[dims.size()-1];



  std::unique_ptr<LSTM> lstm = std::move(NamedLSTM[conv_name]);

  if ((int)C!=(int)lstm->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string((int)C) + ", while the expected input channels of the LSTM are: " + std::to_string(lstm->C);
    LogError(error);
    
    NamedLSTM[conv_name] = std::move(lstm);
    return nullptr;
  }



  tensor_x->Sync();
  tensor_ht->Sync();
  tensor_ct->Sync();

  output = lstm->Forward(tensor_x, tensor_ht, tensor_ct, (int) B, (int)T, thread_id);


  

  int is_forward_func = 1;
  


  std::vector<float> new_dims = {(float)B, (float)lstm->OC}; 



  /*
  Tensor *conv_tensor = NamedTensorsT[conv_name];
  conv_tensor->NewTensor(conv->d_filter, kernel_dims, DimsProd(kernel_dims), true, conv_name);
  conv_tensor->SetIsWeight();
  */

  NamedLSTM[conv_name] = std::move(lstm);
  
  
  //std::cout << "Returning from lstm forward."  << "\n";

  //Tensor *aux = createTensor(nullptr, {}, 0, false, conv_name);

  Tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, "");
  new_tensor->AttrLNode(tensor_x, lstm_op);
  new_tensor->scopeless_name = conv_name;
  return new_tensor;
}















__global__ void embedding_forward_kernel(const float *x, const float *w,
                      float *out, const int tile_size, const int B, const int batches_per_block, const int C, const int OC) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x_block = blockIdx.x;
  int y_block = blockIdx.y;

  
  int col = x_block*tile_size + tx; // OC



  // w e [V, OC]


  for (int i=0; i<batches_per_block; ++i)
  {
    int row = (y_block*batches_per_block+i)*tile_size + ty; // B

    if(row<B && col<OC)
      out[row*OC + col] = w[((int)x[row])*OC + col];
  }
}



void Embedding::SetDescriptors(int B)
{
  this->B=B;
  changed_descriptors=true;
}


float *Embedding::Forward(Tensor *tensor, int B, int thread_id)
{
  float *out = get_from_pool(thread_id, B*OC, "embedding out");


  if (this->B!=B)
    SetDescriptors(B);

  //if(thread_id==0 && nn_mode==training_mode)
  //  NamedTensorsT[Name]->Sparse_Idx_Tensor = tensor;


  int b = B;
  while (b>1 && std::ceil((b*OC)/TILE_SIZE_SQ)>128)
    b-=1;
  int batches_per_block = std::ceil(B/(float)b);



  dim3 block_size(TILE_SIZE, TILE_SIZE);
  dim3 grid_size(std::ceil(OC/(float)TILE_SIZE), std::ceil(b/(float)TILE_SIZE));
  //std::cout << "blocks: " << (grid_size.x*grid_size.y) << ", b " << b << ", B " << B << ", OC " << OC << ", TILE_SIZE " << TILE_SIZE << "\n";
  cudaStream_t stream = ThreadsStream[thread_id];
  embedding_forward_kernel<<<grid_size, block_size, 0, stream>>>(tensor->tensor_ptr, W, out, TILE_SIZE, B, batches_per_block, C, OC);

  return out;
}


__global__ void embedding_backward_kernel(const float *x,
                      float *dw, const float *dy, const int tile_size,
                      int B, int C, int OC) {

  int row = blockIdx.y * blockDim.y + threadIdx.y; // B
  int col = blockIdx.x * blockDim.x + threadIdx.x; // OC

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  

  if(row<B && col<OC)
  {    
    float *_dw = dw + ((int)x[row])*OC + col;
    //float _dy = dy[row*OC + col];

    atomicAdd(_dw, dy[row*OC + col]);
    
    //dw[row*C + col] = tmp;
  }
}

void Embedding::SetBackwardDescriptors()
{

  dW = get_from_pool(0, B*OC, "embedding dW");
  set_to_zero_kernel<<<std::ceil((B*OC)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dW, B*OC);

  changed_descriptors=false;
}

void Embedding::Backward(float *x, float *dy)
{
  /*
  if(changed_descriptors)
    SetBackwardDescriptors();
  //dW = dy;
  copy_tensor_kernel<<<std::ceil((B*OC)/(float)THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, main_stream>>>(dW, dy, B*C);
  */

  

  

  dim3 block_size(TILE_SIZE, TILE_SIZE);
  dim3 grid_size(std::ceil(OC/(float)TILE_SIZE), std::ceil(B/(float)TILE_SIZE));
  embedding_backward_kernel<<<grid_size, block_size, 0, main_stream>>>(x, dW, dy, TILE_SIZE, B, C, OC);
}






extern "C" void *EmbeddingForward(char *self, Tensor *tensor_x, int thread_id, char *conv_namec, int is_obj_attr_or_self)
{
  //TODO: remove self arg and concatenate it instead during the function call
  
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  //std::cout << "Embedding forward of " << conv_name << " with input " << tensor_x->name <<  "\n";



  float *tensor_ptr, *output;
  
  std::vector<float> dims = tensor_x->dims;
  float input_dims_prod = DimsProd(dims);


  std::unique_ptr<Embedding> embedding = std::move(NamedEmbedding[conv_name]);

  tensor_x->Sync();

  
  output = embedding->Forward(tensor_x, tensor_x->dims_prod, thread_id);
  

  int is_forward_func = 1;
  

  std::vector<float> new_dims = tensor_x->dims;
  new_dims.push_back((float)embedding->OC); 

  


  NamedEmbedding[conv_name] = std::move(embedding);
  
  
  Tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, "");
  new_tensor->AttrLNode(tensor_x, embedding_op);
  new_tensor->scopeless_name = conv_name;

  //if(thread_id==0 && nn_mode==training_mode)
  //  new_tensor->Sparse_Idx_Tensor = tensor_x;

  return new_tensor;
}














void BatchNorm2d::SetDescriptors(int H, int W, int B, Tensor *tensor)
{
  this->H = H;
  this->W = W;
  this->B = B;

  /*
  switch(tensor->op)
  {
    case conv2d:
      input_desc = NamedConv2d[tensor->from_cudnn]->output_desc;
      break;
    case bn2drelu:
      input_desc = NamedBN2dRelu[tensor->from_cudnn]->output_desc;
      break;
    case cudnn_relu_op:
      input_desc = NamedRelu[tensor->from_cudnn]->output_desc;
      break;
    case batchnorm2d:
      input_desc = NamedBatchNorm2d[tensor->from_cudnn]->output_desc;
      break;
    case maxpool2d:
      input_desc = NamedMaxPool2d[tensor->from_cudnn]->output_desc;
      break;
    default:
      checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
      checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
      break;
  }*/
  checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
  
  
  checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
  
  checkCUDNN(cudnnCreateTensorDescriptor(&scale_bias_mean_var_desc));
  //checkCUDNN(cudnnSetTensor4dDescriptor(scale_bias_mean_var_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C, 1, 1));
  checkCUDNN(cudnnDeriveBNTensorDescriptor(scale_bias_mean_var_desc, input_desc, CUDNN_BATCHNORM_SPATIAL_PERSISTENT));
}

void BatchNorm2d::InitMovingAverages()
{
  float *aux;

  aux = make_ones_float(C);
  cudaCheck(cudaMalloc(&scale, C*sizeof(float)));
  cudaCheck(cudaMemcpy(scale, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  
  aux = make_zeros_float(C);
  cudaCheck(cudaMalloc(&bias, C*sizeof(float)));
  cudaCheck(cudaMemcpy(bias, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  

  aux = make_zeros_float(C);
  cudaCheck(cudaMalloc(&running_mean, C*sizeof(float)));
  cudaCheck(cudaMemcpy(running_mean, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;

  aux = make_zeros_float(C);
  cudaCheck(cudaMalloc(&saved_mean, C*sizeof(float)));
  cudaCheck(cudaMemcpy(saved_mean, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  
  
  aux = make_ones_float(C);
  cudaCheck(cudaMalloc(&running_var, C*sizeof(float)));
  cudaCheck(cudaMemcpy(running_var, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;

  aux = make_ones_float(C);
  cudaCheck(cudaMalloc(&saved_var, C*sizeof(float)));
  cudaCheck(cudaMemcpy(saved_var, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
}

float *BatchNorm2d::Forward(Tensor *tensor, int H, int W, int B, int C, int thread_id)
{

  if (H != this->H || W != this->W || B != this->B)
    this->SetDescriptors(H, W, B, tensor);

  // Initialize weights.
  if (scale==nullptr)
    this->InitMovingAverages();


  // Forward
  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  
  

  float *output = get_from_pool(thread_id, B * H * W * C, "batchnorm2d");
  //set_to_one_kernel<<<grid_size, block_size>>>(output, B * H * W * C);
  
  
  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  float gamma = 0.9f;
  float eps = 0.00001f;

  

  if(nn_mode==training_mode)
  {
    checkCUDNN(cudnnBatchNormalizationForwardTraining(
      cudnn,
      CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
      &one,
      &zero,
      input_desc,
      tensor->tensor_ptr,
      output_desc,
      output,
      scale_bias_mean_var_desc,
      scale,
      bias,
      gamma,
      running_mean,
      running_var,
      eps,
      saved_mean,
      saved_var
    ));
  }
  else
  {
    checkCUDNN(cudnnDeriveBNTensorDescriptor(scale_bias_mean_var_desc, input_desc, CUDNN_BATCHNORM_SPATIAL));
    checkCUDNN(cudnnBatchNormalizationForwardInference(
      cudnn,
      CUDNN_BATCHNORM_SPATIAL,
      &one,
      &zero,
      input_desc,
      tensor->tensor_ptr,
      output_desc,
      output,
      scale_bias_mean_var_desc,
      scale,
      bias,
      running_mean,
      running_var,
      eps
    ));
  }
  
  return output;
}


extern "C" void *BatchNormForward2d(char *self, Tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self)
{
  //TODO: remove self arg and concatenate it instead during the function call
  //std::cout << "\nBatchNormForward2d " << conv_namec << " and tensor " << tensor.name << "\n";
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  //std::cout << "Conv forward for  conv: " << conv_name <<"\n";
  

  float *tensor_ptr, *output;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float C = dims[dims.size()-3];
  float H = dims[dims.size()-2];
  float W = dims[dims.size()-1];



  std::unique_ptr<BatchNorm2d> conv = std::move(NamedBatchNorm2d[conv_name]);

  if ((int)C!=(int)conv->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string((int)C) + ", while the expected input channels of the BatchNorm2d are: " + std::to_string(conv->C);
    LogError(error);
    
    NamedBatchNorm2d[conv_name] = std::move(conv);
    return nullptr;
  }


  tensor->Sync();
  output = conv->Forward(tensor, H, W, B, C, thread_id);

  float resultingDimsProd = B * (float)C * (float)H * (float)W;

  
  
  std::vector<float> bn_dims = {(float)C};
  std::string bias_name = conv_name+"_bias";

  Tensor *scale_bias_tensor, *scale_tensor, *bias_tensor;

  // for the backprop
  scale_bias_tensor = createTensor(conv->scale, bn_dims, C, true, conv_name);
  scale_bias_tensor->SetBias(conv->bias, C);
  scale_bias_tensor->SetIsWeight();


  // for the optimizer only
  scale_tensor = NamedTensorsT[conv_name];
  scale_tensor->NewTensor(conv->scale, bn_dims, C, true, conv_name);
  scale_tensor->SetIsWeight();
  
  bias_tensor = NamedTensorsT[bias_name];
  bias_tensor->NewTensor(conv->bias, bn_dims, C, true, conv_name);
  bias_tensor->SetIsWeight();



  NamedBatchNorm2d[conv_name] = std::move(conv);

  std::vector<float> new_dims = {(float)B, (float)C, (float)H, (float)W};
  Tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, conv_name);
  new_tensor->AttrNodes(tensor, scale_bias_tensor, batchnorm2d);
  new_tensor->from_cudnn = conv_name;
  return new_tensor;
}



void BatchNorm2d::Backward(float *tensor, float *dx, float *dw, float *db, float *dy)
{
  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  float eps = 0.00001f;
  
  
  checkCUDNN(cudnnBatchNormalizationBackward(
    cudnn,
    CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
    &one,
    &zero,
    &one,
    &one,
    input_desc,
    tensor,
    output_desc,
    dy,
    input_desc,
    dx,
    scale_bias_mean_var_desc,
    scale,
    dw,
    db,
    eps,
    saved_mean,
    saved_var
  ));
}

void batchnormd2d_backward(float *inp, 
                     float *dinp, float *dw, float *db,
                     float *dout, std::string conv_name)
{

  //std::cout << "batchnorm2d_backward for " << conv_name << "\n";
  std::unique_ptr<BatchNorm2d> conv = std::move(NamedBatchNorm2d[conv_name]);

  conv->Backward(inp, dinp, dw, db, dout);

  NamedBatchNorm2d[conv_name] = std::move(conv);

}





void BN2dRelu::SetDescriptors(int H, int W, int B, Tensor *tensor)
{
  //std::cout << "BN2dRelu::SetDescriptors" << "\n";
  /*
  switch(tensor->op)
  {
    case conv2d:
      input_desc = NamedConv2d[tensor->from_cudnn]->output_desc;
      break;
    case bn2drelu:
      input_desc = NamedBN2dRelu[tensor->from_cudnn]->output_desc;
      break;
    case cudnn_relu_op:
      input_desc = NamedRelu[tensor->from_cudnn]->output_desc;
      break;
    case batchnorm2d:
      input_desc = NamedBatchNorm2d[tensor->from_cudnn]->output_desc;
      break;
    case maxpool2d:
      input_desc = NamedMaxPool2d[tensor->from_cudnn]->output_desc;
      break;
    default:
      checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
      checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
      break;
  }*/
  checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));

  
  checkCUDNN(cudnnCreateTensorDescriptor(&intermediate_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(intermediate_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));

  checkCUDNN(cudnnCreateTensorDescriptor(&scale_bias_mean_var_desc));
  //checkCUDNN(cudnnSetTensor4dDescriptor(scale_bias_mean_var_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C, 1, 1));
  checkCUDNN(cudnnDeriveBNTensorDescriptor(scale_bias_mean_var_desc, input_desc, CUDNN_BATCHNORM_SPATIAL_PERSISTENT));  
  
  
  cudnnCreateActivationDescriptor(&activation_desc);
  cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);

  checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
}

void BN2dRelu::InitMovingAverages()
{
  std::cout << "BN2dRelu::InitMovingAverages" << "\n";
  float *aux;

  aux = make_ones_float(C);
  cudaCheck(cudaMalloc(&scale, C*sizeof(float)));
  cudaCheck(cudaMemcpy(scale, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  
  aux = make_zeros_float(C);
  cudaCheck(cudaMalloc(&bias, C*sizeof(float)));
  cudaCheck(cudaMemcpy(bias, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  
  aux = make_zeros_float(C);
  cudaCheck(cudaMalloc(&running_mean, C*sizeof(float)));
  cudaCheck(cudaMemcpy(running_mean, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  
  aux = make_ones_float(C);
  cudaCheck(cudaMalloc(&running_var, C*sizeof(float)));
  cudaCheck(cudaMemcpy(running_var, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  
  aux = make_zeros_float(C);
  cudaCheck(cudaMalloc(&saved_mean, C*sizeof(float)));
  cudaCheck(cudaMemcpy(saved_mean, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
  
  aux = make_ones_float(C);
  cudaCheck(cudaMalloc(&saved_var, C*sizeof(float)));
  cudaCheck(cudaMemcpy(saved_var, aux, C*sizeof(float), cudaMemcpyHostToDevice));
  delete[] aux;
}

float *BN2dRelu::Forward(Tensor *tensor, int H, int W, int B, int C, int thread_id)
{
  std::cout << "BN2dRelu::Forward" << "\n";

  if (H != this->H || W != this->W || B != this->B)
    this->SetDescriptors(H, W, B, tensor);

  // Initialize weights.
  if (scale==nullptr)
    this->InitMovingAverages();


  // Forward
  int grid_size, block_size, shared_mem_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  
  float *intermediate = get_from_pool(thread_id, B * H * W * C, "bn2drelu");
  float *output = get_from_pool(thread_id, B * H * W * C, "bn2drelu");
  //set_to_one_kernel<<<grid_size, block_size>>>(output, B * H * W * C);
  
  
  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  float gamma = 0.1f;
  float eps = 0.00001f;

  

  if(nn_mode==training_mode)
  {
    checkCUDNN(cudnnBatchNormalizationForwardTraining(
      cudnn,
      CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
      &one,
      &zero,
      input_desc,
      tensor->tensor_ptr,
      intermediate_desc,
      intermediate,
      scale_bias_mean_var_desc,
      scale,
      bias,
      gamma,
      running_mean,
      running_var,
      eps,
      saved_mean,
      saved_var
    ));
  }
  else
  {
    checkCUDNN(cudnnBatchNormalizationForwardInference(
      cudnn,
      CUDNN_BATCHNORM_SPATIAL,
      &one,
      &zero,
      input_desc,
      tensor->tensor_ptr,
      intermediate_desc,
      intermediate,
      scale_bias_mean_var_desc,
      scale,
      bias,
      running_mean,
      running_var,
      eps
    ));
  }

  checkCUDNN(cudnnActivationForward(
    cudnn,
    activation_desc,
    &one,
    intermediate_desc,
    intermediate,
    &zero,
    output_desc,
    output
  ));
  
  return output;
}


extern "C" void *BN2dReluForward(char *self, Tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self)
{
  //TODO: remove self arg and concatenate it instead during the function call
  //std::cout << "\nBN2dReluForward2d " << conv_namec << " and tensor " << tensor.name << "\n";
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  std::cout << "Conv forward for conv: " << conv_name <<"\n";
  

  float *tensor_ptr, *output;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float C = dims[dims.size()-3];
  float H = dims[dims.size()-2];
  float W = dims[dims.size()-1];




  std::unique_ptr<BN2dRelu> conv = std::move(NamedBN2dRelu[conv_name]);

  if ((int)C!=(int)conv->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string((int)C) + ", while the expected input channels of the BN2dRelu are: " + std::to_string(conv->C);
    LogError(error);
    
    NamedBN2dRelu[conv_name] = std::move(conv);
    return nullptr;
  }


  tensor->Sync();
  output = conv->Forward(tensor, H, W, B, C, thread_id);

  float resultingDimsProd = B * (float)C * (float)H * (float)W;

  
  
  std::vector<float> bn_dims = {(float)C};
  std::string bias_name = conv_name+"_bias";

  Tensor *scale_bias_tensor, *scale_tensor, *bias_tensor;

  // for the backprop
  scale_bias_tensor = createTensor(conv->scale, bn_dims, C, true, conv_name);
  scale_bias_tensor->SetBias(conv->bias, C);
  scale_bias_tensor->SetIsWeight();


  // for the optimizer only
  scale_tensor = NamedTensorsT[conv_name];
  scale_tensor->NewTensor(conv->scale, bn_dims, C, true, conv_name);
  scale_tensor->SetIsWeight();
  
  bias_tensor = NamedTensorsT[bias_name];
  bias_tensor->NewTensor(conv->bias, bn_dims, C, true, conv_name);
  bias_tensor->SetIsWeight();



  NamedBN2dRelu[conv_name] = std::move(conv);

  std::vector<float> new_dims = {(float)B, (float)C, (float)H, (float)W};
  Tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, conv_name);
  new_tensor->AttrNodes(tensor, scale_bias_tensor, bn2drelu);
  new_tensor->from_cudnn = conv_name;
  return new_tensor;
}



void BN2dRelu::Backward(float *tensor, float *intermediate, float *out, float *dx, float *dw, float *db, float *dintermediate, float *dy)
{
  std::cout << "BN2dRelu::Backward" << "\n";
  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  float eps = 0.00001f;
  
  
  checkCUDNN(cudnnBatchNormalizationBackward(
    cudnn,
    CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
    &one,
    &zero,
    &one,
    &one,
    input_desc,
    tensor,
    intermediate_desc,
    dintermediate,
    input_desc,
    dx,
    scale_bias_mean_var_desc,
    scale,
    dw,
    db,
    eps,
    saved_mean,
    saved_var
  ));


  checkCUDNN(cudnnActivationBackward(
                        cudnn,
                        activation_desc,
                        &one,
                        output_desc,
                        out,
                        output_desc,
                        dy,
                        intermediate_desc,
                        intermediate,
                        &zero,
                        intermediate_desc,
                        dintermediate
  ));
}

void bn2drelu_backward(float *inp, float *intermediate, float *out,
                     float *dinp, float *dw, float *db, float *dintermediate,
                     float *dout, std::string conv_name)
{

  //std::cout << "batchnorm2d_backward for " << conv_name << "\n";
  std::unique_ptr<BN2dRelu> conv = std::move(NamedBN2dRelu[conv_name]);

  conv->Backward(inp, intermediate, out, dinp, dw, db, dintermediate, dout);

  NamedBN2dRelu[conv_name] = std::move(conv);

}








void Relu::SetDescriptors(int C, int H, int W, int B, Tensor *tensor)
{
  this->C = C;
  this->H = H;
  this->W = W;
  this->B = B;

  
  switch(tensor->op)
  {
    case conv2d:
      input_desc = NamedConv2d[tensor->from_cudnn]->output_desc;
      break;
    case bn2drelu:
      input_desc = NamedBN2dRelu[tensor->from_cudnn]->output_desc;
      break;
    case cudnn_relu_op:
      input_desc = NamedRelu[tensor->from_cudnn]->output_desc;
      break;
    case batchnorm2d:
      input_desc = NamedBatchNorm2d[tensor->from_cudnn]->output_desc;
      break;
    case maxpool2d:
      input_desc = NamedMaxPool2d[tensor->from_cudnn]->output_desc;
      break;
    default:
      checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
      checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
      break;
  }

  
  cudnnCreateActivationDescriptor(&activation_desc);
  cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);

  checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
}


float *Relu::Forward(Tensor *tensor, int H, int W, int B, int C)
{

  if (H != this->H || W != this->W || B != this->B)
    this->SetDescriptors(C, H, W, B, tensor);



  float *output = get_from_pool(0, B * H * W * C, "Relu");
  //set_to_one_kernel<<<grid_size, block_size>>>(output, B * H * W * C);
  
  
  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  



  checkCUDNN(cudnnActivationForward(
    cudnn,
    activation_desc,
    &one,
    input_desc,
    tensor->tensor_ptr,
    &zero,
    output_desc,
    output
  ));
  
  return output;
}


extern "C" void *ReluForward(char *self, Tensor *tensor, char *conv_namec, int is_obj_attr_or_self)
{
  
  //TODO: remove self arg and concatenate it instead during the function call
  //std::cout << "\nReluForward2d " << conv_namec << " and tensor " << tensor.name << "\n";
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;


  float *tensor_ptr, *output;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float C = dims[dims.size()-3];
  float H = dims[dims.size()-2];
  float W = dims[dims.size()-1];



  std::unique_ptr<Relu> conv = std::move(NamedRelu[conv_name]);


  tensor->Sync();
  output = conv->Forward(tensor, H, W, B, C);

  float resultingDimsProd = B * (float)C * (float)H * (float)W;

  
  
  std::vector<float> bn_dims = {(float)C};
  std::string bias_name = conv_name+"_bias";

  Tensor *scale_bias_tensor, *scale_tensor, *bias_tensor;




  NamedRelu[conv_name] = std::move(conv);

  std::vector<float> new_dims = {(float)B, (float)C, (float)H, (float)W};
  Tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, conv_name);
  new_tensor->AttrLNode(tensor, cudnn_relu_op);
  new_tensor->from_cudnn = conv_name;
  return new_tensor;
}



void Relu::Backward(float *tensor, float *out, float *dx, float *dy)
{
  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  float eps = 0.00001f;
  
  

  checkCUDNN(cudnnActivationBackward(
                        cudnn,
                        activation_desc,
                        &one,
                        output_desc,
                        out,
                        output_desc,
                        dy,
                        input_desc,
                        tensor,
                        &zero,
                        input_desc,
                        dx
  ));
}

void cudnn_relu_backward(float *inp, float *out,
                     float *dinp, 
                     float *dout, std::string conv_name)
{

  //std::cout << "batchnorm2d_backward for " << conv_name << "\n";
  std::unique_ptr<Relu> conv = std::move(NamedRelu[conv_name]);

  conv->Backward(inp, out, dinp, dout);

  NamedRelu[conv_name] = std::move(conv);

}








void Conv2d::SetDescriptors(int H, int W, int B, Tensor *tensor)
{
  this->H = H;
  this->W = W;
  this->B = B;


  //std::cout << "\nConv2d Set Descriptors\nC: " << C << " OC " << OC << " ks " << ks << " stride " << stride << " padding " << padding << " H " << H << " W " << W << "\n";


  out_H = std::floor((H - ks + 2 * padding) / stride) + 1;
  out_W = std::floor((W - ks + 2 * padding) / stride) + 1;
  //std::cout << "Out H: " << out_H << " out W: " << out_W << "\n";


  /*
  switch(tensor->op)
  {
    case conv2d:
      input_desc = NamedConv2d[tensor->from_cudnn]->output_desc;
      break;
    case bn2drelu:
      input_desc = NamedBN2dRelu[tensor->from_cudnn]->output_desc;
      break;
    case cudnn_relu_op:
      input_desc = NamedRelu[tensor->from_cudnn]->output_desc;
      break;
    case batchnorm2d:
      input_desc = NamedBatchNorm2d[tensor->from_cudnn]->output_desc;
      break;
    case maxpool2d:
      input_desc = NamedMaxPool2d[tensor->from_cudnn]->output_desc;
      break;
    default:
      // Initialize input tensor descriptor
      checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
      checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
      break;
  }*/

  checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
  
  // Initialize filter descriptor
  cudnnFilterDescriptor_t filter_desc;
  checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc));
  checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OC, C, ks, ks));
  this->filter_desc = filter_desc;

  // Initialize convolution descriptor
  cudnnConvolutionDescriptor_t conv_desc;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
  checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1,
                                           CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
  this->conv_desc = conv_desc;

  // Initialize output tensor descriptor
  cudnnTensorDescriptor_t output_desc;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, OC, out_H, out_W));
  this->output_desc = output_desc;

  
  int requested_algo_count;
  int algo_count;




  // Forward
  checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &requested_algo_count));
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(requested_algo_count);
  checkCUDNN(cudnnFindConvolutionForwardAlgorithm(
        cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        requested_algo_count,
        &algo_count,
        perf_results.data()
  ));

  this->fwd_algo = perf_results.front().algo;


  
  if (d_workspace!=nullptr)
    move_to_pool_pow2(0, workspace_size, d_workspace, "d workspace");
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        fwd_algo,
        &workspace_size
  ));
  d_workspace = get_from_pool_pow2(0, workspace_size, "d workspace");
  
  




  // Backward to input
  checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnn, &requested_algo_count));
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results_back_y(requested_algo_count);
  checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithm(
        cudnn,
        filter_desc,
        output_desc,
        conv_desc,
        input_desc,
        requested_algo_count,
        &algo_count,
        perf_results_back_y.data()
  ));

  y_bwd_algo = perf_results_back_y.front().algo;

  
  if(d_workspace_y_back!=nullptr)
    move_to_pool_pow2(0, workspace_size_y_back, d_workspace_y_back, "d workspace y back");
  checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn,
        filter_desc,
        output_desc,
        conv_desc,
        input_desc,
        y_bwd_algo,
        &workspace_size_y_back
  ));

  d_workspace_y_back = get_from_pool_pow2(0, workspace_size_y_back, "d workspace y back");
  




  // Backward to weight
  checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnn, &requested_algo_count));
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results_back_w(requested_algo_count);
  checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(
        cudnn,
        input_desc,
        output_desc,
        conv_desc,
        filter_desc,
        requested_algo_count,
        &algo_count,
        perf_results_back_w.data()
  ));

  w_bwd_algo = perf_results_back_w.front().algo;

  
  
  if (d_workspace_w_back!=nullptr)
    move_to_pool_pow2(0, workspace_size_w_back, d_workspace_w_back, "conv d workspace w back");
  checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn,
        input_desc,
        output_desc,
        conv_desc,
        filter_desc,
        w_bwd_algo,
        &workspace_size_w_back
  ));
  d_workspace_w_back = get_from_pool_pow2(0, workspace_size_w_back, "conv d workspace w back");
  
  
}





void Conv2d::InitFilters()
{
  std::vector<float> h_filter;
  float *filter;
  for (std::size_t idx = 0; idx < C * OC; ++idx) {

    if (Init=="xavu_relu")
      filter = make_xavier_uniform_float_relu(ks*ks, ks*ks*C, ks*ks*OC);
    if (Init == "xavu_tanh")
      filter = make_xavier_uniform_float_tanh(ks*ks, ks*ks*C, ks*ks*OC);
    if (Init=="he_normal_relu")
      filter = make_he_normal_float_relu(ks*ks, ks*ks*C);
    if (Init == "init_gpt")
      filter = make_gpt_init(ks*ks);
    if (Init=="xavu")
      filter = make_xavier_uniform_float(ks*ks, ks*ks*C, ks*ks*OC);
    if (Init=="zeros")
      filter = make_zeros_float(ks*ks);
    if (Init=="ones")
      filter = make_ones_float(ks*ks);
    if (Init=="randu")
      filter = make_random_float_uniform(ks*ks);


    for (int i=0; i < ks*ks; i++)
      h_filter.emplace_back(filter[i]);

    delete[] filter;
    //for (const auto& val : filter) 
    //  h_filter.emplace_back(val);
  }
    
  float* d_filter = nullptr;
  const std::size_t filter_size = h_filter.size();
  cudaCheck(cudaMalloc(&d_filter, filter_size * sizeof(float)));

  cudaCheck(cudaMemcpy(d_filter, h_filter.data(), filter_size * sizeof(float), cudaMemcpyDefault));
  this->d_filter = d_filter;
  
}





float *Conv2d::Forward(Tensor *tensor, int H, int W, int B, int thread_id)
{
  // Initialize descriptors.
  //std::cout << "\nConv2d Forward with H: " << H << " W: " << W << "\n";

  if (H != this->H || W != this->W || B != this->B)
    this->SetDescriptors(H, W, B, tensor);

  // Initialize weights.
  if (d_filter==nullptr)
    this->InitFilters();


  
  // Forward
  float *d_output = get_from_pool(thread_id, B * out_H * out_W * OC, "conv2d");

  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;

  
 

  checkCUDNN(cudnnConvolutionForward(
        cudnn,
        &one,
        input_desc,
        tensor->tensor_ptr,
        filter_desc,
        d_filter,
        conv_desc,
        fwd_algo,
        d_workspace,
        workspace_size,
        &zero,
        output_desc,
        d_output
    ));
  



  return d_output;
}


void Conv2d::Backward(float *tensor, float *dx, float *d_filter_g, float *dy)
{
  //std::cout << "\nConv2d Backward with H: " << H << " W: " << W << "\n";


  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  
  // Backward to input
  checkCUDNN(cudnnConvolutionBackwardData(
    cudnn,
    &one,
    filter_desc, // input tensor descriptor
    d_filter,
    output_desc, // output grad tensor descriptor
    dy,
    conv_desc, // convolution descriptor
    y_bwd_algo, //Obtained with getConvolutionBackwardDataAlgorithm
    d_workspace_y_back, 
    workspace_size_y_back, //Obtained with getConvolutionBackwardDataWorkspaceSize
    &zero,
    input_desc, // filter descriptor
    dx
  ));


  // Backward to weight
  checkCUDNN(cudnnConvolutionBackwardFilter(
    cudnn,
    &one,
    input_desc, // input tensor descriptor
    tensor,
    output_desc, // output grad tensor descriptor
    dy,
    conv_desc, // convolution descriptor
    w_bwd_algo, //Obtained with getConvolutionBackwardFilterAlgorithm
    d_workspace_w_back, 
    workspace_size_w_back, //Obtained with getConvolutionBackwardFilterWorkspaceSize
    &one,
    filter_desc, // filter descriptor
    d_filter_g
  ));


  /*
  std::cout << "d_w is:\n";
  PrintTensorF(d_filter_g, C*OC, ks*ks);
  std::cout << "\n";
  */

}



void conv2d_backward(float *inp,  float *weight,
                     float *dinp, float *dw,
                     float *dout, std::string conv_name)
{

  //std::cout << "conv2d_backward for " << conv_name << "\n";
  std::unique_ptr<Conv2d> conv = std::move(NamedConv2d[conv_name]);

  conv->Backward(inp, dinp, dw, dout);


  NamedConv2d[conv_name] = std::move(conv);

  
}




extern "C" void *ConvForward2d(char *self, Tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self)
{
  //TODO: remove self arg and concatenate it instead during the function call
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  //std::cout << "Conv forward of " << conv_name << " and tensor " << tensor->name << "\n";
  //std::cout << "Conv forward for  conv: " << conv_name <<"\n";
  

  float *tensor_ptr, *output, *d_filter;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float C = dims[dims.size()-3];
  float H = dims[dims.size()-2];
  float W = dims[dims.size()-1];



  std::unique_ptr<Conv2d> conv = std::move(NamedConv2d[conv_name]);



  if ((int)C!=(int)conv->C)
  {
    std::string error = "Input tensor channels are: " + std::to_string((int)C) + ", while the expected input channels of the convolution are: " + std::to_string(conv->C);
    LogError(error);
    
    NamedConv2d[conv_name] = std::move(conv);
    return nullptr;
  }



  tensor->Sync();

  output = conv->Forward(tensor, H, W, B, thread_id);

  int ks_H = conv->ks;
  int ks_W = conv->ks;


  
  
  float resultingDimsProd = B * (float)conv->OC * (float)conv->out_H * (float)conv->out_W;

  int is_forward_func = 1;
  


  std::vector<float> new_dims = {(float)conv->B, (float)conv->OC, (float)conv->out_H, (float)conv->out_W};
  

  //for backprop:
  std::vector<float> kernel_dims = {(float)conv->OC, (float)C, (float)conv->ks, (float)conv->ks}; 




  Tensor *conv_tensor = NamedTensorsT[conv_name];
  conv_tensor->NewTensor(conv->d_filter, kernel_dims, DimsProd(kernel_dims), true, conv_name);
  conv_tensor->SetIsWeight();
  

  NamedConv2d[conv_name] = std::move(conv);

  Tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, "");
  new_tensor->AttrNodes(tensor, conv_tensor, conv2d);
  new_tensor->from_cudnn = conv_name;
  return new_tensor;
}





extern "C" float CreateConv2dOnDemand(char *tensor_name, char *init,
                                      float C, float OC, float ks, float stride, float padding)
{
  std::cout << "\nCreate conv on demand:\n   C: " << C << " OC " << OC << " ks " << ks << " stride " << stride << " padding " << padding << "\n";

  auto conv = std::make_unique<Conv2d>((int)C, (int)OC, (int)ks, (int)stride, (int)padding, init, tensor_name);

  std::cout << "Adding " << tensor_name << " to NamedConv2d dict\n";
  NamedConv2d[tensor_name] = std::move(conv);
  return 0;
}




extern "C" float CreateBatchNorm2dOnDemand(char *tensor_name, float C)
{
  std::cout << "\nCreate BatchNorm2d " << tensor_name << " on demand:\n   C: " << C  << "\n";

  auto conv = std::make_unique<BatchNorm2d>((int)C, tensor_name);

  NamedBatchNorm2d[tensor_name] = std::move(conv);
  return 0;
}


extern "C" float CreateBN2dReluOnDemand(char *tensor_name, float C)
{
  std::cout << "\nCreate BatchNorm2d on demand:\n   C: " << C  << "\n";

  auto conv = std::make_unique<BN2dRelu>((int)C, tensor_name);

  NamedBN2dRelu[tensor_name] = std::move(conv);
  return 0;
}


extern "C" float CreateLSTMOnDemand(char *tensor_name, char *init,
                                      float C, float OC)
{
  std::cout << "\nCreate lstm on demand:\n   C: " << C << " OC " << OC << "\n";

  auto lstm = std::make_unique<LSTM>((int)C, (int)OC, init, tensor_name);

  std::cout << "Adding " << tensor_name << " to NamedLSTM dict\n";
  NamedLSTM[tensor_name] = std::move(lstm);
  return 0;
}

extern "C" float CreateEmbeddingOnDemand(char *tensor_name, char *init,
                                      float C, float OC)
{
  std::cout << "\nCreate embedding on demand:\n   C: " << C << " OC " << OC << "\n";

  auto embedding = std::make_unique<Embedding>((int)C, (int)OC, init, tensor_name);

  std::cout << "Adding " << tensor_name << " to NamedEmbedding dict\n";
  NamedEmbedding[tensor_name] = std::move(embedding);
  return 0;
}



extern "C" float CreateLinearOnDemand(char *tensor_name, char *init,
                                      float C, float OC, int_vec *Notators)
{
  std::cout << "\nCreate linear on demand:\n  C " << C << " OC " << OC << "\n";
  std::cout << "" << tensor_name << " " << init << "\n";


  std::unique_ptr<Linear> linear = std::make_unique<Linear>((int) C, int (OC), init, Notators, tensor_name);

  std::cout << "Adding " << tensor_name << " to NamedMHSA dict\n";
  NamedLinear[tensor_name] = std::move(linear);
  return 0;
}




extern "C" float CreateMHSAOnDemand(char *tensor_name, char *init,
                                      float nh, float C, float T, int_vec *notators)
{
  std::cout << "\nCreate mhsa on demand:\n   nh: " << nh << " C " << C << " T " << T << "\n";
  std::cout << "" << tensor_name << " " << init << "\n";

  std::unique_ptr<MHSA> mhsa = std::make_unique<MHSA>((int)nh, (int)T, (int) C, init, notators, tensor_name);

  std::cout << "Adding " << tensor_name << " to NamedMHSA dict\n";
  NamedMHSA[tensor_name] = std::move(mhsa);
  return 0;
}



extern "C" float CreateReluOnDemand(char *tensor_name)
{
  auto conv = std::make_unique<Relu>(tensor_name);

  NamedRelu[tensor_name] = std::move(conv);
  return 0;
}






void MaxPool2d::SetDescriptors(int H, int W, int B, int C, Tensor *tensor)
{
  this->H = H;
  this->W = W;
  this->B = B;


  out_H = std::floor((H - ks + 2 * padding) / stride) + 1;
  out_W = std::floor((W - ks + 2 * padding) / stride) + 1;

  this->out_H=out_H;
  this->out_W=out_W;
  //std::cout << "Out H: " << out_H << " out W: " << out_W << "\n";

  /*
  switch(tensor->op)
  {
    case conv2d:
      input_desc = NamedConv2d[tensor->from_cudnn]->output_desc;
      break;
    case bn2drelu:
      input_desc = NamedBN2dRelu[tensor->from_cudnn]->output_desc;
      break;
    case cudnn_relu_op:
      input_desc = NamedRelu[tensor->from_cudnn]->output_desc;
      break;
    case batchnorm2d:
      input_desc = NamedBatchNorm2d[tensor->from_cudnn]->output_desc;
      break;
    case maxpool2d:
      input_desc = NamedMaxPool2d[tensor->from_cudnn]->output_desc;
      break;
    default:
      checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
      checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
      break;
  }*/
  checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
  


  // Initialize pooling descriptor
  checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
  checkCUDNN(cudnnSetPooling2dDescriptor(pooling_desc,            //descriptor handle
                                         (Type=="max") ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,       //mode - max pooling
                                         CUDNN_NOT_PROPAGATE_NAN, //NaN propagation mode
                                         ks,                       //window height
                                         ks,                       //window width
                                         padding,                       //vertical padding
                                         padding,                       //horizontal padding
                                         stride,                       //vertical stride
                                         stride));                     //horizontal stride
  
  // Initialize output tensor descriptor
  checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, out_H, out_W));
}



float *MaxPool2d::Forward(Tensor *tensor, int H, int W, int B, int C, int thread_id)
{
  // Initialize descriptors.
  //std::cout << "\nPool2d Forward with H: " << H << " W: " << W << "\n";
  //std::cout << "Type: " << Type << "\n";


  if (H != this->H || W != this->W || B != this->B || C != this->C)
    this->SetDescriptors(H, W, B, C, tensor);


  
  // Forward
  float *d_output = get_from_pool(thread_id, B * out_H * out_W * C, "maxpool2d");
  

  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;

  checkCUDNN(cudnnPoolingForward(
        cudnn,
        pooling_desc,
        &one,
        input_desc,
        tensor->tensor_ptr,
        &zero,
        output_desc,
        d_output
    ));
  

  return d_output;
}


extern "C" void *MaxPoolForward2d(char *self, Tensor *tensor, int thread_id, char *conv_namec, int is_obj_attr_or_self)
{
  //std::cout << "MaxPoolForward2d of " << conv_namec << " and tensor " << tensor.name << "\n";
  
  std::string _self = self;
  std::string conv_name = conv_namec;
  if (is_obj_attr_or_self)
    conv_name = _self + conv_name;

  //std::cout << "Conv forward for  conv: " << conv_name <<"\n";
  

  float *tensor_ptr, *output, *d_filter;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float input_dims_prod = DimsProd(dims);

  float B = dims[0];
  float C = dims[dims.size()-3];
  float H = dims[dims.size()-2];
  float W = dims[dims.size()-1];
  float OC = C;


  std::unique_ptr<MaxPool2d> conv = std::move(NamedMaxPool2d[conv_name]);

  tensor->Sync();
  output = conv->Forward(tensor, H, W, B, C, thread_id);


  
  
  float resultingDimsProd = B * (float)OC * (float)conv->out_W * (float)conv->out_W;



  std::vector<float> new_dims = {(float)B, (float)OC, (float)conv->out_H, (float)conv->out_W};
  

  NamedMaxPool2d[conv_name] = std::move(conv);

  Tensor *new_tensor = createTensor(output, new_dims, DimsProd(new_dims), false, conv_name);
  new_tensor->AttrLNode(tensor, maxpool2d);
  new_tensor->from_cudnn = conv_name;
  return new_tensor;
}


void MaxPool2d::Backward(float *tensor, float *out, float *dx, float *dy)
{
  //std::cout << "\nMaxPool2d Backward with H: " << H << " W: " << W << "\n";


  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;


  // Backward to input
  checkCUDNN(cudnnPoolingBackward(
    cudnn,
    pooling_desc,
    &one,
    output_desc,
    out,
    output_desc, // output grad tensor descriptor
    dy,
    input_desc,
    tensor,
    &zero,
    input_desc, // input descriptor
    dx
  ));
  
}


void maxpool2d_backward(float *inp,  float *out,
                     float *dinp,
                     float *dout, std::string conv_name)
{
  //std::cout << "maxpool2d_backward of " << conv_name << "\n";
  std::unique_ptr<MaxPool2d> conv = std::move(NamedMaxPool2d[conv_name]);

  conv->Backward(inp, out, dinp, dout);

  NamedMaxPool2d[conv_name] = std::move(conv);

  
}


extern "C" float CreateMaxPool2dOnDemand(char *tensor_name, char *type, float ks, float stride, float padding)
{
  std::cout << "\nCreate maxpool2d on demand:\n" << "   ks " << ks << " stride " << stride << " padding " << padding << "\n";

  auto maxpool = std::make_unique<MaxPool2d>((int)ks, (int)stride, (int)padding, type);

  NamedMaxPool2d[tensor_name] = std::move(maxpool);
  return 0;
}





unsigned long long time_seed() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto seed = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    return seed;
}



__global__ void random_padding_cropping_kernel(
  const float* input,
  float* output,
  int batch_size,
  int channels,
  int input_height,
  int input_width,
  int crop_height,
  int crop_width,
  int padding,
  unsigned long long seed)
{
  int b = blockIdx.x;
  int c = blockIdx.y;
  int hw = blockIdx.z * blockDim.x + threadIdx.x;

  int y = hw / input_width;
  int x = hw % input_width;


  //int y = threadIdx.y;
  //int x = threadIdx.x;

  if (b >= batch_size || c >= channels || y >= crop_height || x >= crop_width)
    return;

  // Setup random number generator
  curandState state;
  curand_init(seed, b, 0, &state); // one random state per batch

  // Generate random padding values
  int pad_top = curand(&state) % (padding + 1); // Range belongs to [0, padding]
  int pad_bottom = curand(&state) % (padding + 1);
  int pad_left = curand(&state) % (padding + 1);
  int pad_right = curand(&state) % (padding + 1);

  // Compute the padded height and width
  int padded_height = input_height + pad_top + pad_bottom;
  int padded_width = input_width + pad_left + pad_right;

  // Ensure crop dimensions fit within padded dimensions
  int max_crop_top = padded_height - crop_height;
  int max_crop_left = padded_width - crop_width;

  // Generate random crop starting points
  int crop_top = curand(&state) % (max_crop_top + 1);
  int crop_left = curand(&state) % (max_crop_left + 1);

  // Compute input coordinates for any location within the crop
  int input_y = crop_top + y;
  int input_x = crop_left + x;

  // Handle padding and out-of-bounds conditions
  if (input_y >= pad_top && input_y < padded_height - pad_bottom &&
      input_x >= pad_left && input_x < padded_width - pad_right) {
    // Within padded region, read from input
    output[b * channels * crop_height * crop_width + c * crop_height * crop_width + y * crop_width + x] =
      input[b * channels * input_height * input_width + c * input_height * input_width + (input_y - pad_top) * input_width + (input_x - pad_left)];
  } else {
    // Outside padded region, set to zero
    output[b * channels * crop_height * crop_width + c * crop_height * crop_width + y * crop_width + x] = 0.0f;
  }
}


extern "C" void *RandomCrop(int thread_id, Tensor *tensor, float padding)
{
  float *tensor_ptr, *cropped;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float dims_prod = tensor->dims_prod;

  unsigned long long seed = get_int_seed();

  float B, C, H, W;
  B = dims[0];
  C = dims[dims.size()-3];
  H = dims[dims.size()-2];
  W = dims[dims.size()-1];

  cropped = get_from_pool(thread_id, dims_prod, "cropping");


  int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

  dim3 numBlocks(B, C, std::ceil((H*W)/(float)block_size));
  dim3 threadsPerBlock(block_size);
  cudaCheck(cudaGetLastError());


  random_padding_cropping_kernel<<<numBlocks, threadsPerBlock, 0, tensor->cuda_stream>>>(
    tensor_ptr,
    cropped,
    B,
    C,
    H,
    W,
    H,
    W,
    padding,
    seed
  );
  cudaCheck(cudaGetLastError());
  
  Tensor *new_tensor = createTensor(cropped, dims, dims_prod, false, "", tensor->cuda_stream);
  new_tensor->AttrLNode(tensor, crop_op);
  return new_tensor;
}



__global__ void random_horizontal_flip_kernel(const float *input_tensor, float *output_tensor, 
                                              int batch_size, int channels, int height, int width, 
                                              unsigned long long seed) {
  // Thread ID
  int batch_idx = blockIdx.x;
  int channel_idx = blockIdx.y;
  int hw = blockIdx.z * blockDim.x + threadIdx.x;

  int h = hw / width;
  int w = hw % width;

  int idx = ((batch_idx * channels + channel_idx) * height + h) * width + w;

  if (idx > (batch_size*channels*height*width))
    return;

  // Random number generator initialization
  curandState state;
  curand_init(seed, batch_idx, 0, &state);
    

  // Determine if the current image should be flipped
  bool flip = curand_uniform(&state) > 0.5f;

  if (flip) {
    // Compute the flipped column index
    int flipped_col_idx = width - w - 1;
    int flipped_idx = ((batch_idx * channels + channel_idx) * height + h) * width + flipped_col_idx;
    output_tensor[idx] = input_tensor[flipped_idx];
  } else {
    output_tensor[idx] = input_tensor[idx];
  }
}



extern "C" void *RandomHorizontalFlip(int thread_id, Tensor *tensor)
{
  float *tensor_ptr, *flipped;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float dims_prod = tensor->dims_prod;

  unsigned long long seed = get_int_seed();

  float B, C, H, W;
  B = dims[0];
  C = dims[dims.size()-3];
  H = dims[dims.size()-2];
  W = dims[dims.size()-1];

  flipped = get_from_pool(thread_id, dims_prod, "horizontal_flipping");


  int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

  dim3 numBlocks(B, C, std::ceil((H*W)/(float)block_size));
  dim3 threadsPerBlock(block_size);
  cudaCheck(cudaGetLastError());

  //std::cout << "B " << B << ", C " << C << ", H " << H << ", W " << W << "\n";

  random_horizontal_flip_kernel<<<numBlocks, threadsPerBlock, 0, tensor->cuda_stream>>>(
    tensor_ptr,
    flipped,
    B,
    C,
    H,
    W,
    seed
  );
  cudaCheck(cudaGetLastError());
  
  Tensor *new_tensor = createTensor(flipped, dims, dims_prod, false, "", tensor->cuda_stream);
  new_tensor->AttrLNode(tensor, random_horizontal_flip_op);
  return new_tensor;
}



__global__ void normalize_img_kernel(float *output_tensor, const float *input_tensor, 
                                     const float *mean, const float *std,
                                     int batch_size, int channels, int height, int width) {
  // Thread ID
  int b = blockIdx.x;
  int c = blockIdx.y;
  int hw = blockIdx.z * blockDim.x + threadIdx.x;
  
  int h = hw / width;
  int w = hw % width;

  //int idx = ((batch_idx * channels + c) * height + h) * width + w;
  int idx = b * channels * height * width + c * height * width + h * width + w;
    
  output_tensor[idx] = (input_tensor[idx]-mean[c])/std[c];    
}


extern "C" void *NormalizeImg(int thread_id, Tensor *tensor, Tensor *mean, Tensor *std)
{
  float *tensor_ptr, *normalized;
  tensor_ptr = tensor->tensor_ptr;
  std::vector<float> dims = tensor->dims;
  float dims_prod = tensor->dims_prod;


  float B, C, H, W;
  B = dims[0];
  C = dims[dims.size()-3];
  H = dims[dims.size()-2];
  W = dims[dims.size()-1];

  if(mean->dims_prod!=C||std->dims_prod!=C)
  { 
    LogErrorS("NormalizeImg mean and std tensors must have the same dimensionality as the image channels.");
    return nullptr;
  }

  normalized = get_from_pool(thread_id, dims_prod, "normalize img");



  int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

  dim3 numBlocks(B, C, std::ceil((H*W)/(float)block_size));
  dim3 threadsPerBlock(block_size);
  cudaCheck(cudaGetLastError());

  normalize_img_kernel<<<numBlocks, threadsPerBlock, 0, tensor->cuda_stream>>>(
    normalized,
    tensor_ptr,
    mean->tensor_ptr,
    std->tensor_ptr,
    B,
    C,
    H,
    W
  );

  cudaCheck(cudaGetLastError());

  Tensor *new_tensor = createTensor(normalized, dims, dims_prod, false, "", tensor->cuda_stream);
  new_tensor->AttrLNode(tensor, normalize_img_op);
  return new_tensor;
}

__global__ void jitter_kernel(float *y, const float *x, const float factor, const int dims_prod,
                               unsigned long long seed)
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx>dims_prod)
    return;


  curandState state;
  curand_init(seed, idx, 0, &state);

  float r = curand_normal(&state);
  r = cuda_clip(r, -2.0, 2.0);

  y[idx] = x[idx] * (1+factor*r);
}

extern "C" void *Jitter(int thread_id, Tensor *tensor, float factor)
{


  int grid_size, block_size;
  float dims_prod = tensor->dims_prod;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];


  float *jittered = get_from_pool(thread_id, dims_prod, "jitter img");
  unsigned long long seed = get_int_seed();
  jitter_kernel<<<grid_size, block_size, 0, tensor->cuda_stream>>>(jittered, tensor->tensor_ptr, factor, dims_prod, seed);


  Tensor *new_tensor = createTensor(jittered, tensor->dims, dims_prod, false, "", tensor->cuda_stream);
  new_tensor->AttrLNode(tensor, jitter_op);
  return new_tensor;
}



__global__ void scalarmult_backward_kernel(float *dx, const float *dy,
                                           const float scalar,
                                           int dims_prod) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dims_prod)
    {
      dx[i] = scalar * dy[i];
    }
}
void scalarmult_backward(float *dx, float *dy, float scalar, float dims_prod)
{
  //std::cout << "scalar mult backward with scalar " << scalar <<  "\n";
  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  scalarmult_backward_kernel<<<grid_size, block_size, 0, main_stream>>>(dx, dy, scalar, dims_prod);
}


__global__ void hadamard_backward_kernel(const float *x, const float *w,
                                         float *dx, float *dw, const float *dy,
                                         int dims_prod) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dims_prod)
    {
      dx[i] = w[i] * dy[i];
      dw[i] = x[i] * dy[i];
    }
}

void hadamard_backward(float *x, float *w, float *dx, float *dw, float *dy, float dims_prod)
{
  //std::cout << "hadamard_backward" <<  "\n";
  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  hadamard_backward_kernel<<<grid_size, block_size, 0, main_stream>>>(x, w, dx, dw, dy, dims_prod);
}


__global__ void dropout_mask_kernel(float *y, float *m, const float *x, float rate, float scale,
                               int dims_prod,
                               unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dims_prod)
    {
      curandState state;
      curand_init(seed, i, 0, &state);

      float r = curand_uniform(&state);
      if(r<rate)
        m[i] = 0;
      else
        m[i] = scale;
      
      y[i] = m[i]*x[i];
    }
}

extern "C" void *dropout(int thread_id, Tensor *tensor, float rate)
{
  if (nn_mode==training_mode&&thread_id==0)
  {
    float dims_prod = tensor->dims_prod;

    int grid_size, block_size;
    std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
    grid_size = grid_block_mem_sizes[0];
    block_size = grid_block_mem_sizes[1];

    float *dropout_ptr = get_from_pool(thread_id, dims_prod, "dropout forward");
    float *device_y = get_from_pool(thread_id, dims_prod, "dropout forward output");

    float scale = 1 / (1-rate);
    
    unsigned long long seed = get_int_seed();

    dropout_mask_kernel<<<grid_size, block_size, 0, main_stream>>>(device_y, dropout_ptr, tensor->tensor_ptr, rate, scale, dims_prod, seed);
    
    Tensor *dropout_tensor = createTensor(dropout_ptr, tensor->dims, dims_prod, true, "");
    dropout_tensor->scopeless_name="";

    Tensor *new_tensor = createTensor(device_y, tensor->dims, dims_prod, false, "");
    new_tensor->AttrNodes(tensor, dropout_tensor, dropout_op);
    return new_tensor;
  }
  return tensor;
}


__global__ void dropout_backward_kernel(float *dx, float *m, const float *dy,
                               int dims_prod) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dims_prod)
      dx[i] = m[i]*dy[i];
}
void dropout_backward(float *dx, float *mask, float *dy, float dims_prod)
{
  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  dropout_backward_kernel<<<grid_size, block_size, 0, main_stream>>>(dx, mask, dy, dims_prod);
}



// Parallelizes over B, C
__global__ void crossentropy_softmax_backward_kernel1(float* dlogits,
                           const float* probs, const float* targets,
                           int B, int C, float scale) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = threadIdx.x;
    
    
    if (i < B * C) {
        int b = i / (C);
        int v = i % C;

        float *dlogits_b = dlogits + b * C;
        const float *probs_b = probs + b * C;

        //float ix = targets[v];
        float ix = targets[b * C + v]; // one-hot tensor
        float p = probs_b[v];

        //float indicator = (v==ix) ? 1.0f : 0.0f; // one-hot already
        float indicator = ix;

        dlogits_b[v] += (p - indicator) * scale;
        
    }
}


void CrossEntropyBackward(float *y_hat,
                          float *y,
                          int B, int C, 
                          float *dloss,
                          float scale)
{
  
  /*
  int grid_size = B;
  int block_size = 32;
  size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
  */
  

  float *probs = get_from_pool(0, B*C,"ce probs");

  //int grid_size, block_size;
  //size_t shared_mem_size;
  

  int grid_size, block_size, shared_mem_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size  = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  set_to_zero_kernel<<<grid_size, block_size, 0, main_stream>>>(probs, B*C);


  /*
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*32*C);
  grid_size  = B*32;
  block_size = grid_block_mem_sizes[1];
  
  online_softmax<<<grid_size, block_size, 0, main_stream>>>(y_hat, probs, B, C);
  */
  
  
  

  
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size  = B;
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = 2 * block_size / 32 * sizeof(float);

  softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size, main_stream>>>(y_hat, probs, B, C);
  
  


  
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  
  crossentropy_softmax_backward_kernel1<<<grid_size, block_size, 0, main_stream>>>(dloss, probs, y, B, C, scale);
  move_to_pool(0, B*C, probs,"ce probs");

  
}



extern "C" float cross_entropy(Tensor *y_hat, Tensor *y, float scale)
{
  
  Tensor *loss_tensor = new Tensor();


  loss_tensor->AttrNodes(y_hat, y, cross_entropy_op);
  loss_tensor->scalar = scale;


  todo_backward_tensors.push_back(loss_tensor);

  

  return 0;
}




__global__ void crossentropy_idx_backward_kernel(float* dlogits,
                           const float* probs, const float* targets,
                           int B, int C, float scale) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int i = threadIdx.x;
    
    
    if (i < B * C) {
        int b = i / (C);
        int v = i % C;

        float *dlogits_b = dlogits + b * C;
        const float *probs_b = probs + b * C;


        float p = probs_b[v];

        float indicator = (v==targets[b]) ? 1.0f : 0.0f;
        //float indicator = ix;

        dlogits_b[v] += (p - indicator) * scale;
        
    }
}



void CrossEntropyIdxBackward(float *y_hat,
                          float *y,
                          int B, int C, 
                          float *dloss,
                          float scale)
{
  float *probs = get_from_pool(0, B*C,"ce probs");


  int grid_size, block_size, shared_mem_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size  = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  set_to_zero_kernel<<<grid_size, block_size, 0, main_stream>>>(probs, B*C);


  

  /*
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size  = B;
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = 2 * block_size / 32 * sizeof(float);

  softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size, main_stream>>>(y_hat, probs, B, C);
  */
  grid_block_mem_sizes = CalculateSimpleWarpGridAndBlockSizes(B);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  online_softmax<<<grid_size, block_size, 0, main_stream>>>(y_hat, probs, B, C);
  
  


  
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  
  crossentropy_idx_backward_kernel<<<grid_size, block_size, 0, main_stream>>>(dloss, probs, y, B, C, scale);
  move_to_pool(0, B*C, probs,"ce probs");
}



extern "C" float cross_entropy_idx(Tensor *y_hat, Tensor *y, float scale)
{
  
  Tensor *loss_tensor = new Tensor();


  loss_tensor->AttrNodes(y_hat, y, cross_entropy_idx_op);
  loss_tensor->scalar = scale;


  todo_backward_tensors.push_back(loss_tensor);

  

  return 0;
}





__global__ void mse_kernel(float *dy, const float* y_hat, const float* y,
                            const float scale, const float dims_prod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < dims_prod) {
        dy[idx] = 2 * (y_hat[idx] - y[idx]) * scale;
    }
}

void MSEBackward(float *y_hat, float *y,
                 int dims_prod, 
                 float *dloss,
                 float scale)
{
  //std::cout << "MSE Backward" << "\n";

  int grid_size, block_size, shared_mem_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size  = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  
  mse_kernel<<<grid_size, block_size, 0, main_stream>>>(dloss, y_hat, y, scale, dims_prod);

  //PrintTensorF(dloss, 1, dims_prod);
}

extern "C" float mse(Tensor *y_hat, Tensor *y, float scale)
{  
  Tensor *loss_tensor = new Tensor();


  loss_tensor->AttrNodes(y_hat, y, mse_op);
  loss_tensor->scalar = scale;


  todo_backward_tensors.push_back(loss_tensor);


  return 0;
}



__global__ void online_mse(float *out, const float *y_hat, const float *y_true, int N, int C) {
  
    const int warpsPerBlock = blockDim.x / warpSize;
    int tid = threadIdx.x;

    

    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    // one warp one row
    int row = blockIdx.x * warpsPerBlock + warpId;
    
    if (laneId >= C)
        return;

    if (row >= N)
        return;

    
    const float *x = y_hat  + row * C;
    const float *y = y_true + row * C;
    

    // merge calculating maxval and sumval in one loop
    // which is an arithmetic improvment from online softmax over normal softmax
    float sumval = 0.0f;

#pragma unroll
    for (int i = laneId; i < C; i += warpSize) {
        // when updating the maxval, dynamically updates the previous sumval by
        // multiplying e^{previous_maxval - current_maxval}

        sumval += powf(x[i]-y[i], 2);
    }

    // use warp functions instead of cooperative groups for better readibility
    // calculate the warp wised maxval and sumval
    float offsetSumval;

#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        __syncwarp();
        offsetSumval = __shfl_down_sync(0xFFFFFFFF, sumval, offset);
        
        sumval += offsetSumval;
    }


    sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);


    out[row] = sumval/C;
    
}


extern "C" void *mse_with_priorities(int thread_id, Tensor *y_hat, Tensor *y, float scale, Tensor *is_w)
{  
  Tensor *mse_tensor, *loss_tensor;
  mse_tensor = new Tensor();
  loss_tensor = new Tensor();



  
  mse_tensor->AttrNodes(y_hat, y, lgrad_op);
  mse_tensor->scalar = scale;
  mse_tensor->dims = y_hat->dims;
  mse_tensor->dims_prod = y_hat->dims_prod;

  loss_tensor->AttrNodes(mse_tensor, is_w, mse_is_w_op);
  

  //loss_tensor->AttrNodes(y_hat, y, mse_op);



  todo_backward_tensors.push_back(loss_tensor);


  std::vector<float> dims = format_BatchFirst_Dims(y_hat->dims);
  float B = dims[0];
  float C = dims[1];


  float *msed = get_from_pool(0, B, "mse with priorities");

  int grid_size, block_size;
  std::vector<int> grid_block_mem_sizes = CalculateSimpleWarpGridAndBlockSizes(B);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  

  online_mse<<<grid_size, block_size, 0, main_stream>>>(msed, y_hat->tensor_ptr, y->tensor_ptr, B, C);

  Tensor *new_tensor = createTensor(msed, {B}, B, false, "");
  new_tensor->AttrLNode(y_hat, detach_op);
  return new_tensor;
}

__global__ void mse_with_priorities_kernel(float *dy, const float* y_hat, const float* y, const float *is_w,
                            const float scale, const float dims_prod) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dims_prod) {
        dy[tid] = 2 * (y_hat[tid] - y[tid]) * scale * is_w[tid];
    }
}


void MSEWithPrioritiesBackward(Tensor *loss_tensor,
                 float *dloss)
{
  //std::cout << "MSEWithPriorities Backward" << "\n";

  
  Tensor *y_hat_tensor, *y_tensor, *is_w_tensor;
  y_hat_tensor = loss_tensor->L_Node->L_Node;
  y_tensor = loss_tensor->L_Node->R_Node;
  is_w_tensor = loss_tensor->R_Node;
  float scale = loss_tensor->L_Node->scalar;

  int dims_prod = y_hat_tensor->dims_prod;

  int grid_size, block_size, shared_mem_size;
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
  grid_size  = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];
  
  //std::cout << "grid_size: " << grid_size << ", block_size: " << block_size << "\n";
  mse_with_priorities_kernel<<<grid_size, block_size, 0, main_stream>>>(dloss, y_hat_tensor->tensor_ptr, y_tensor->tensor_ptr, is_w_tensor->tensor_ptr, scale, dims_prod);

  //PrintTensorF(dloss, 1, dims_prod);
}


//




//




//









std::vector<std::string> scopes;


int DoesTreeContainWeight(Tensor *back_node)
{
  if(back_node==nullptr)
    return 0;
  
  if(back_node->weight)
    return 1;

  //if(in_int(back_node->op, activation_ops))
  //  return 1;
  
  return DoesTreeContainWeight(back_node->L_Node) + DoesTreeContainWeight(back_node->R_Node);
}

void ThreadedCleanupToPool(Tensor *back_node, std::string scope, int thread_id)
{
  if(back_node==nullptr||back_node->weight)
    return;
  //std::cout << "-----Clean threaeded " << back_node->name << "\n";
  

  
  if (!in_str(scope, scopes));
    scopes.push_back(scope);

  ThreadedCleanupToPool(back_node->L_Node, scope, thread_id);
  ThreadedCleanupToPool(back_node->R_Node, scope, thread_id);

  
  to_pool_threaded(back_node->dims_prod, back_node->tensor_ptr, scope, thread_id, "");
  to_free_tensor_threaded(back_node, scope, thread_id);
}

void CleanThreadTensors(std::string scope, int thread_id)
{
  for(Tensor *tensor : threaded_Tensors_to_free[thread_id][scope])
    delete tensor;

  std::vector<float*> scope_tensors_ptrs;
  

  for(std::tuple<float, float *, std::string> pair : threaded_tensors_to_pool[thread_id][scope])
    move_to_pool(thread_id, std::get<0>(pair), std::get<1>(pair), std::get<2>(pair));


  threaded_Tensors_to_free[thread_id][scope].clear();
  threaded_tensors_to_pool[thread_id][scope].clear();
  threaded_tensors_sent_to_pool[thread_id][scope].clear();

  threaded_Tensors_to_free[thread_id].erase(scope);
  threaded_tensors_to_pool[thread_id].erase(scope);
  threaded_tensors_sent_to_pool[thread_id].erase(scope);
}

void ForwardCleanupToPool(Tensor *back_node, std::string scope)
{
  if(back_node==nullptr||back_node->weight)
    return;
  

  
  if (!in_str(scope, scopes));
    scopes.push_back(scope);

  ForwardCleanupToPool(back_node->L_Node, scope);
  ForwardCleanupToPool(back_node->R_Node, scope);

  
  to_pool_forward(back_node->dims_prod, back_node->tensor_ptr, scope, "");
  to_free_tensor_forward(back_node, scope);
}

void CleanScopeTensors(std::string scope)
{
  for(Tensor *tensor : forward_Tensors_to_free[scope])
    delete tensor;

  std::vector<float*> scope_tensors_ptrs;

  //for(auto &pair : scope_tensors[scope])
  //  scope_tensors_ptrs.push_back(pair.second);

  for(std::tuple<float, float *, std::string> pair : forward_tensors_to_pool[scope])
  {
    //if(!in_float_ptr_vec(std::get<1>(pair), scope_tensors_ptrs))
      move_to_pool(0, std::get<0>(pair), std::get<1>(pair), std::get<2>(pair));

    //move_to_pool(pair.first, pair.second);
    //cudaCheck(cudaFree(std::get<1>(pair)));
  }

  forward_Tensors_to_free[scope].clear();
  forward_tensors_to_pool[scope].clear();
  forward_tensors_sent_to_pool[scope].clear();

  forward_Tensors_to_free.erase(scope);
  forward_tensors_to_pool.erase(scope);
  forward_tensors_sent_to_pool.erase(scope);
}

extern "C" float clean_forward(char *scope, char *previous_scope, int thread_id, int has_grad)
{//TODO: break? clears threaded tensors
  for(std::string _scope : scopes)
    CleanScopeTensors(_scope);
  scopes.clear();
  
  cudaCheck(cudaGetLastError());
  return 0;
}


void TraversePreOrder(Tensor *back_node, float *device_dy, bool from_gradless, bool from_custom, int parent_op)
{
  if(back_node==nullptr)
    return;

  int op=back_node->op;
  std::string tensor_name, param_name, bias_name;
  float *w;
  float *device_dx, *device_dw;
  device_dx=nullptr;
  device_dw=nullptr;
  float dims_prod = back_node->dims_prod;

  

  if(!in_int(op, gradless_ops) && !from_gradless)
  {

    //std::cout << "\nTraversing: " << back_node->name << "/" << back_node->scopeless_name << ", op: " << back_node->op << ", parent_op: " << parent_op << ", leaf: " << back_node->leaf << ", weight: " << back_node->weight << "\n";
    if(device_dy==nullptr && !in_int(op, loss_ops) && !from_custom)
    {
      std::string _err = "dy derivate is null at the backward mode with op "+std::to_string(op);
      LogErrorS(_err);
      return;
    }



    if (back_node->weight) // dw is updated by pointer
      return;
    

    tensor_name = back_node->scopeless_name;
    if (back_node->leaf)
    {
      if (!from_custom)
      {
        if(tensor_name!="")
        {
          if(var_to_grad.count(tensor_name)>0)
          {
            
            float *acc_y = var_to_grad[tensor_name];
            
            int grid_size, block_size, shared_mem_size;
            std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
            grid_size = grid_block_mem_sizes[0];
            block_size = grid_block_mem_sizes[1];

            
            add_inplace<<<grid_size, block_size>>>(acc_y, device_dy, dims_prod);

            to_pool(dims_prod, acc_y, "dy of leaf");

          } else
            var_to_grad[tensor_name] = device_dy;
        }
        to_pool(dims_prod, device_dy, "dy of leaf");
      }
      //std::cout << "\n\nAccumulating grad of: " << tensor_name << "\n\n\n";
      
      to_pool(dims_prod, back_node->tensor_ptr, "leaf tensor");
      
      to_free_tensor(back_node);
      return;
    }

    from_custom = from_custom || (in_int(op, custom_ops));

    int B, C, OC;
    float x_size, w_size, b_size;

    float *inp, *b, *out, *last_inp;
    float *dinp, *dw, *db, *device_db;
    device_dw=nullptr;
    device_db=nullptr;
    w=nullptr;
    b=nullptr;
    

    
    //std::cout << "Acquire info"  << "\n";

    tensor_name = back_node->L_Node->scopeless_name;

    inp = back_node->L_Node->tensor_ptr;
    x_size = back_node->L_Node->dims_prod;

    out = back_node->tensor_ptr;

    //std::cout << "Check null" << "\n";
    if(back_node->R_Node!=nullptr)
    {
      //std::cout << "not null " << "\n";
      param_name  = back_node->R_Node->name;
      w = back_node->R_Node->tensor_ptr;
      w_size = back_node->R_Node->dims_prod;

      b = back_node->R_Node->b;
      b_size = back_node->R_Node->b_size;
    }


    //std::cout << "malloc device w" << "\n";

    // weight gradient
    if(!in_int(op, loss_ops)&&back_node->R_Node!=nullptr)
    {
      
      int grid_size, block_size; 
      std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(w_size);
      grid_size = grid_block_mem_sizes[0];
      block_size = grid_block_mem_sizes[1];
      
      //std::cout << "Is weight: " << back_node->R_Node->weight << "\n";
      if(back_node->R_Node->weight)
      {
        float *new_grad_ptr;
        if (w!=nullptr&&op!=hadamard_op&&op!=add_op)
        {
          //std::cout << "weight of size " << w_size << "\n";
          if (NamedParamGrads[param_name]==nullptr)
          {
            
            new_grad_ptr = get_from_pool(0, w_size, "weight grad pointer");
            set_to_zero_kernel<<<grid_size, block_size, 0, main_stream>>>(new_grad_ptr, w_size);
            NamedParamGrads[param_name] = new_grad_ptr;
          }
          device_dw = NamedParamGrads[param_name];
        }

        if (b!=nullptr&&op!=hadamard_op&&op!=add_op)
        {
          bias_name = param_name+"_bias";

          if (NamedParamGrads[bias_name]==nullptr)
          {
            int grid_size_b, block_size_b; 
            std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(w_size);
            grid_size_b = grid_block_mem_sizes[0];
            block_size_b = grid_block_mem_sizes[1];
            
            new_grad_ptr = get_from_pool(0, w_size, "bias grad pointer");
            set_to_zero_kernel<<<grid_size, block_size, 0, main_stream>>>(new_grad_ptr, b_size);
            NamedParamGrads[bias_name] = new_grad_ptr;
          }
          device_db = NamedParamGrads[bias_name];
        }
      } else {
      
        if(!in_int(op, weightless_ops) && !from_custom && back_node->R_Node->op != detach_op)
        {
          /*
          if (w_size==4)
          {
            std::cout << "ulululu of op " << std::to_string(op) << "\n";
            std::cout << "" << param_name<< "\n";
            std::cout << "" << tensor_name<< "\n";
          }
          */
          std::string from = "dw of " + std::to_string(op);
          device_dw = get_from_pool(0, w_size, from);
          set_to_zero_kernel<<<grid_size, block_size, 0, main_stream>>>(device_dw, w_size);
        }
      }
    }
    


    // input gradient
    std::string from = "dx of "+ std::to_string(op);
    

    if(op!=add_op && op!=scalar_add_op && !from_custom && op!=lgrad_op && op!=broadcast_lastdim_add_op) {
      int grid_size, block_size; 
      std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(x_size);
      grid_size = grid_block_mem_sizes[0];
      block_size = grid_block_mem_sizes[1];

      device_dx = get_from_pool(0, x_size, from);

      //TODO: remove this set to zero to improve performance (then, adjust gather op dx to be set to zero)
      set_to_zero_kernel<<<grid_size, block_size, 0, main_stream>>>(device_dx, x_size);
    }
    

    //std::cout << "malloc done"  << "\n";


    B=0;
    C=0;
    OC=0;
    if(back_node->L_Node->dims.size()>0)
    {
      std::vector<float> BC = format_LinearLayer_Dims(back_node->L_Node->dims);
      B  = BC[0];
      C  = BC[1];
    }
    if (w!=nullptr)
      OC = back_node->R_Node->dims[0];
    


    //std::cout << "EXECUTING OP  " << op << "\n";
    switch (op)
    {
      // Simple Leaf Nodes Ops
      case scalar_add_op:
        device_dx = device_dy;
        break;
      case scalar_mult_op:
        scalarmult_backward(device_dx, device_dy, back_node->scalar, x_size); //todo: This one may be wrong
        break;
      case mult_op:
        matmul_backward(inp, w, B, C, OC, device_dx, device_dw, device_dy);
        break;
      case conv2d:
        conv2d_backward(inp, w, device_dx, device_dw, device_dy, param_name);
        break;
      case maxpool2d:
        maxpool2d_backward(inp, out, device_dx, device_dy, back_node->name);
        break;
      case batchnorm2d:
        batchnormd2d_backward(inp, device_dx, device_dw, device_db, device_dy, back_node->name);
        break;
      //case bn2drelu:
      //  bn2drelu_backward(inp, intermediate, out, device_dx, device_dw, device_db, device_dintermediate, device_dy, back_node->name);
      //  break;
      case relu_op:
        relu_backward(inp, x_size, device_dx, device_dy);
        break;
      case cudnn_relu_op:
        cudnn_relu_backward(inp, out, device_dx, device_dy, back_node->name);
        break;
      case gelu_op:
        gelu_backward(inp, x_size, device_dx, device_dy);
        break;
      case sigmoid_op:
        sigmoid_backward(out, x_size, device_dx, device_dy);
        break;
      case tanh_op:
        tanh_backward(out, x_size, device_dx, device_dy);
        break;
      case add_op:
        device_dx = device_dy;
        device_dw = device_dy;
        break;
      case hadamard_op:
        hadamard_backward(inp, w, device_dx, device_dw, device_dy, x_size);
        break;
      case dropout_op:
        dropout_backward(device_dx, w, device_dy, x_size);
        device_dw = device_dy;
        break;
      case gather_last_dim_op:
        gather_last_dim_backward(device_dx, device_dy, back_node);
        device_dw = device_dy;
        break;
      case broadcast_lastdim_add_op:
        device_dx = device_dy;
        broadcast_lastdim_add_backward(device_dw, device_dy, w_size, x_size);
        break;
      case mean_over_semilast_dim_op:
        mean_over_semilast_dim_backward(device_dx, device_dy, back_node);
        break;
      
      // Custom Ops
      case sigmoid_add2weights_op:
        sigmoid_add2weights_backward(back_node, device_dy);
        device_dx = device_dy;
        device_dw = device_dy;
        break;
      case lstm_op:
        lstm_backward(inp, device_dx, device_dy, back_node->scopeless_name);
        break;
      case embedding_op:
        embedding_backward(inp, device_dy, back_node->scopeless_name);
        break;
      case mhsa_op:
        mhsa_backward(inp, device_dx, device_dy, back_node->scopeless_name);
        break;
      case linear_op:
        linear_backward(inp, device_dx, device_dy, back_node->scopeless_name);
        break;

      // Loss Ops
      case cross_entropy_op:
        CrossEntropyBackward(inp, w, B, C, device_dx, back_node->scalar);
        break;
      case cross_entropy_idx_op:
        CrossEntropyIdxBackward(inp, w, B, C, device_dx, back_node->scalar);
        break;
      case mse_op:
        MSEBackward(inp, w, back_node->L_Node->dims_prod, device_dx, back_node->scalar);
        break;
      case mse_is_w_op:
        MSEWithPrioritiesBackward(back_node, device_dx);
        break;

      case lgrad_op:
        device_dx = device_dy;
        break;

      default:
        std::string _error = "The operation "+std::to_string(op)+" does not yet have a backward implementation";
        LogErrorS(_error);
        break;
    }

    //if (ends_with(tensor_name, "ht"))
    //  PrintTensorF(device_dx, 4, 256);
  
  } else
  {
    //std::cout << "\n\nFROM A GRADLESS OP" << "\n\n\n";
    from_gradless = true;
  }

  
  if (in_int(op, loss_ops)||op==lgrad_op)
  {
    to_pool(back_node->R_Node->dims_prod, back_node->R_Node->tensor_ptr, "in loss_ops");
    delete back_node->R_Node;
    back_node->R_Node = nullptr;
  }
  


  // Garbage Collector on all lines below
  TraversePreOrder(back_node->L_Node, device_dx, from_gradless, from_custom, op);
  //from_gradless = (from_gradless || in_int(op, loss_ops));
  TraversePreOrder(back_node->R_Node, device_dw, from_gradless, from_custom, op);
  


  if (back_node->Sparse_Idx_Tensor!=nullptr)
    save_from_pool(back_node->Sparse_Idx_Tensor);

  
  if(!in_int(op, loss_ops) && back_node->tensor_ptr!=nullptr) //loss op has leaves only
    to_pool(dims_prod, back_node->tensor_ptr, "op tensor");


  std::string _op = "dy of operation " + std::to_string(op) + " from parent op " + std::to_string(parent_op) + " and parameter " + param_name;  
  if(device_dy!=nullptr)
    to_pool(dims_prod, device_dy, _op);

  if (!back_node->weight)
    to_free_tensor(back_node);
}


extern "C" float backprop()
{

  int op;
  
  std::string tensor_name;
  
  float *device_dy=nullptr;



  while(todo_backward_tensors.size()>0)
  {
    //std::cout << "\n\nbackprop:\n\n\n";
    Tensor *back_node = todo_backward_tensors.back();
    todo_backward_tensors.pop_back();

    to_free_tensor(back_node);

    op = back_node->op;
    
    if (op==attribution)
    {
      tensor_name = back_node->name;
      //std::cout << "\n\n\n   backward attribution of " << tensor_name << "\n";
      device_dy = var_to_grad[tensor_name];
      //if (device_dy==nullptr)
      //  std::cout << "propagating null device_dy"  << "\n";
      var_to_grad.erase(tensor_name);
      
      back_node = back_node->R_Node;
    }

    
    TraversePreOrder(back_node, device_dy, false, false, op);
  }





  for(Tensor *tensor : backprop_Tensors_to_save) // e.g: sparse idx tensors
  {
    
    backprop_Tensors_to_free.erase(std::remove(backprop_Tensors_to_free.begin(), backprop_Tensors_to_free.end(), tensor), backprop_Tensors_to_free.end());
    
    for(std::tuple<float, float *, std::string> pair : backprop_tensors_to_pool)
    {
      float *tensor_ptr = std::get<1>(pair);
      if (tensor->tensor_ptr == tensor_ptr)
      {
        //std::cout << "Remove " << tensor->name << "/" << tensor->scopeless_name << " from pool.\n";
        backprop_tensors_to_pool.erase(std::remove(backprop_tensors_to_pool.begin(), backprop_tensors_to_pool.end(), pair), backprop_tensors_to_pool.end());
        break;
      }
    }
    
  }

  for(Tensor *tensor : backprop_Tensors_to_free)
    delete tensor;

  for(std::tuple<float, float *, std::string> pair : backprop_tensors_to_pool)
  {
    move_to_pool(0, std::get<0>(pair), std::get<1>(pair), std::get<2>(pair));
    //move_to_pool(0, pair.first, pair.second);
  }

  backprop_Tensors_to_save.clear();
  backprop_Tensors_to_free.clear();
  backprop_tensors_to_pool.clear();
  tensors_sent_to_pool.clear();
  var_to_grad.clear();
  return 0;
}





class Optimizer {
public:
  virtual ~Optimizer() = default;
  std::map<std::string, float *> NamedV, NamedM;

  int timestep = 1;
  float lr = 0.0f;
  //float eps = 1.5e-4;
  float eps = 1e-8;
    
  virtual void init_states(std::string, float) {}
  virtual void step(float *, float *, std::vector<float>, std::string, cudaStream_t) {}
  virtual void sparse_step(float *, float *, float *, std::vector<float>, std::vector<float>, std::string, cudaStream_t) {}
  virtual void count_step() {
    timestep+=1;
  }
};

class SGD_optim : public Optimizer {
  float lr, momentum, weight_decay, grad_clip;

  public:
    SGD_optim(float lr, float momentum, float weight_decay, float grad_clip)
      : lr(lr), momentum(momentum), weight_decay(weight_decay), grad_clip(grad_clip) {}
    
  void init_states(std::string param_name, float params_count) override;
  void step(float *param, float *grad, std::vector<float> dims, std::string param_name, cudaStream_t stream) override;
  void sparse_step(float *, float *, float *, std::vector<float>, std::vector<float> dims, std::string param_name, cudaStream_t stream) override;
};

class AdamW_optim : public Optimizer {
  float lr, beta1, beta2, weight_decay, grad_clip;

  public:
    AdamW_optim(float lr, float beta1, float beta2, float weight_decay, float grad_clip)
      : lr(lr), beta1(beta1), beta2(beta2), weight_decay(weight_decay), grad_clip(grad_clip) {}
    
  void init_states(std::string param_name, float params_count) override;
  void step(float *param, float *grad, std::vector<float> dims, std::string param_name, cudaStream_t stream) override;
  void sparse_step(float *, float *, float *, std::vector<float>, std::vector<float> dims, std::string param_name, cudaStream_t stream) override;
};



__device__ inline float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

__global__ void sgd_kernel(float* params_memory, const float* grads_memory, float* m_memory, long num_parameters,
                              float learning_rate, float momentum,
                              const float weight_decay, const float grad_clip) {

   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= num_parameters) return;  // guard
   
   float grad = std::clamp(grads_memory[i], -grad_clip, grad_clip) + weight_decay * params_memory[i];
   float m = m_memory[i];
   
   // update the first moment (momentum)
   m = m*momentum + grad;
   m_memory[i] = m;
  
   params_memory[i] -= learning_rate * m;
}

__global__ void adamw_kernel(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction,
                              const float eps, const float weight_decay, const float grad_clip) {

   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= num_parameters) return;  // guard
   
   float grad = std::clamp(grads_memory[i], -grad_clip, grad_clip);
   float m = m_memory[i];
   float v = v_memory[i];
   // update the first moment (momentum)
   m = lerp(grad, m, beta1);
   m_memory[i] = m;
   // update the second moment (RMSprop)
   v = lerp(grad * grad, v, beta2);
   v_memory[i] = v;
   m /= beta1_correction;  // m_hat
   v /= beta2_correction;  // v_hat
   params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}


void SGD_optim::init_states(std::string param_name, float params_count)
{
  

  if (NamedM[param_name]==nullptr)
  {
    std::cout << "init_states for param " << param_name << " with params count: " << params_count << "\n";

    float *m, *device_m;

    m = new float[params_count];
    m = make_zeros_float(params_count);

    cudaMalloc(&device_m, params_count*sizeof(float));
    cudaMemcpy(device_m, m, params_count*sizeof(float), cudaMemcpyHostToDevice);

    delete[] m;

    NamedM[param_name] = device_m;
  }
}

void SGD_optim::step(float *param, float *grad, std::vector<float> dims, std::string param_name, cudaStream_t stream)
{
  float *m = NamedM[param_name];

 
  int params_count = DimsProd(dims);
  int grid_size, block_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(params_count);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  sgd_kernel<<<grid_size, block_size, 0, stream>>>(param, grad, m, params_count,
                                           lr, momentum, weight_decay, grad_clip);
}

void SGD_optim::sparse_step(float *param, float *grad, float *idx, std::vector<float> idx_dims, std::vector<float> dims, std::string param_name, cudaStream_t stream)
{
  float *m = NamedM[param_name];
}

void AdamW_optim::init_states(std::string param_name, float params_count)
{
  if (NamedV[param_name]==nullptr)
  {
    std::cout << "init_states for param " << param_name << " with params count: " << params_count << "\n";

    float *v, *m, *device_v, *device_m;
    v = new float[params_count];
    m = new float[params_count];

    v = make_zeros_float(params_count);
    m = make_zeros_float(params_count);


    cudaMalloc(&device_v, params_count*sizeof(float));
    cudaMalloc(&device_m, params_count*sizeof(float));
    cudaMemcpy(device_v, v, params_count*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_m, m, params_count*sizeof(float), cudaMemcpyHostToDevice);

    delete[] v;
    delete[] m;

    NamedV[param_name] = device_v; 
    NamedM[param_name] = device_m;
  }
}

void AdamW_optim::step(float *param, float *grad, std::vector<float> dims, std::string param_name, cudaStream_t stream)
{
  float *v = NamedV[param_name];
  float *m = NamedM[param_name];

  float beta1_correction = 1.0f - powf(beta1, timestep);
  float beta2_correction = 1.0f - powf(beta2, timestep);


  int params_count = DimsProd(dims);
  
  int grid_size, block_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(params_count);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  adamw_kernel<<<grid_size, block_size, 0, stream>>>(param, grad, m, v, params_count,
                                           lr, beta1, beta2, beta1_correction, beta2_correction,
                                           eps, weight_decay, grad_clip);
}


__global__ void sparse_adamw_kernel(float* params_memory, const float* grads_memory, const float *idx_tensor,
                              float* m_memory, float* v_memory, long num_parameters, const int C,
                              const float learning_rate, const float beta1, const float beta2, const float beta1_correction, const float beta2_correction,
                              const float eps, const float weight_decay, const float grad_clip) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_parameters) return;  // guard

  int b = i / C;
  int c = i % C;


  int idx = (int)idx_tensor[b]; 


  float grad = std::clamp(grads_memory[i], -grad_clip, grad_clip);
  float m = m_memory[idx*C + c];
  float v = v_memory[idx*C + c];
  // update the first moment (momentum)
  m = lerp(grad, m, beta1);
  // update the second moment (RMSprop)
  v = lerp(grad * grad, v, beta2);
  
  m /= beta1_correction;  // m_hat
  v /= beta2_correction;  // v_hat

  //float *param = params_memory + idx*C + c;
  float p = params_memory[idx*C + c];

  p = p - learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
  //atomicAdd(param, -1*(learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i])));
  __threadfence();
  m_memory[idx*C + c] = m;
  v_memory[idx*C + c] = v;
  params_memory[idx*C + c] = p;
}

void AdamW_optim::sparse_step(float *param, float *grad, float *idx, std::vector<float> idx_dims, std::vector<float> dims, std::string param_name, cudaStream_t stream)
{
  float *v = NamedV[param_name];
  float *m = NamedM[param_name];

  float beta1_correction = 1.0f - powf(beta1, timestep);
  float beta2_correction = 1.0f - powf(beta2, timestep);


  int leading_dim = dims[dims.size()-1];

  int params_count = DimsProd(idx_dims)*leading_dim;

  
  int grid_size, block_size; 
  std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(params_count);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  //std::cout << "Sparse step " << "\n";
  //PrintDims(idx_dims);
  //PrintDims(dims);
  sparse_adamw_kernel<<<grid_size, block_size, 0, stream>>>(param, grad, idx, m, v, params_count, leading_dim,
                                           lr, beta1, beta2, beta1_correction, beta2_correction,
                                           eps, weight_decay, grad_clip);
}



std::unique_ptr<Optimizer> optimize(std::unique_ptr<Optimizer> optimizer)
{
  int num_streams = NamedParamGrads.size();

  std::vector<cudaStream_t> streams(num_streams);

  for (int i = 0; i < num_streams; ++i)
  {

    cudaStreamCreate(&streams[i]);
    //StreamAwaitStreamB(streams[i], main_stream);
  }

  cudaStreamSynchronize(main_stream);

  int i=0;
  for (auto& pair : NamedParamGrads)
  {
    std::string param_name = pair.first;
    //std::cout << "Optimizing " << param_name << "\n";

    if (param_name!="none")
    {
      float *grad = pair.second;
      Tensor *tensor = NamedTensorsT[param_name];
      
      //std::cout << "param dims: "  << "\n";
      //PrintDims(tensor->dims);
      optimizer->init_states(param_name, tensor->dims_prod);

      if (tensor->Sparse_Idx_Tensor!=nullptr)
      {
        //std::cout << "Tensor " << param_name << " has a sparse gradient "<< "\n";
        Tensor *idx_tensor = tensor->Sparse_Idx_Tensor;

        optimizer->sparse_step(tensor->tensor_ptr, grad, idx_tensor->tensor_ptr,
                               idx_tensor->dims, tensor->dims, param_name, streams[i]);

        move_to_pool(0, idx_tensor->dims_prod, idx_tensor->tensor_ptr, "sparse grad idxs");
        delete idx_tensor;
      } else
        optimizer->step(tensor->tensor_ptr, grad, tensor->dims, param_name, streams[i]);

      int grid_size, block_size; 
      std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(tensor->dims_prod);
      grid_size = grid_block_mem_sizes[0];
      block_size = grid_block_mem_sizes[1];

      set_to_zero_kernel<<<grid_size, block_size, 0, streams[i]>>>(grad, tensor->dims_prod);
    }
    i+=1;
  }
  optimizer->count_step();

  
  for (int i = 0; i < num_streams; ++i)
  {
    cudaStreamSynchronize(streams[i]);
    //StreamAwaitStreamB(main_stream, streams[i]);
  }
  for (int i = 0; i < num_streams; ++i)
    cudaStreamDestroy(streams[i]);

  cudaStreamSynchronize(main_stream);

  return std::move(optimizer);
}



std::unique_ptr<Optimizer> optimizer = nullptr;
extern "C" float SGD(float lr, float momentum, float weight_decay, float grad_clip)
{

  if (optimizer==nullptr)
    optimizer = std::make_unique<SGD_optim>(lr, momentum, weight_decay, grad_clip);

  optimizer = optimize(std::move(optimizer));

  return 0;
}
extern "C" float AdamW(float lr, float beta1, float beta2, float weight_decay, float grad_clip)
{

  if (optimizer==nullptr)
    optimizer = std::make_unique<AdamW_optim>(lr, beta1, beta2, weight_decay, grad_clip);

  optimizer = optimize(std::move(optimizer));

  return 0;
}





extern "C" float OneCycleLR(float base_lr, float step, float max_steps)
{
  // Possibly wrong.
  float pct_start, final_div_factor, max_momentum, min_momentum, cycle_length, down_phase_steps, min_lr;
  pct_start=0.3;
  final_div_factor=1000;
  max_momentum=0.95;
  min_momentum=0.85;

  cycle_length = int(max_steps*pct_start);
  down_phase_steps = max_steps - cycle_length;

  min_lr = base_lr/final_div_factor;

  
  if(step<cycle_length)
    return base_lr * step/cycle_length;
  if(step<max_steps)
    return base_lr * (1 + std::cos(M_PI * (step - cycle_length) / (max_steps - cycle_length))) / 2;
  return min_lr;
}

extern "C" float CosineLR(float base_lr, float min_lr, float step, float max_steps)
{
  //float min_lr = base_lr*0.05;

  if(step<max_steps)
    return min_lr + (base_lr-min_lr) * (1 + std::cos(M_PI * (step/max_steps))) / 2;
  return min_lr;
}





extern "C" float eval()
{
  std::cout << "\n\n\nSETTING NN MODE TO EVAL" << "\n\n";
  
  /*
  for(int i=0; i<TensorPool.size(); i++)
  {
    for(int j=0; j<TensorPool[i].size();j++)
      cudaCheck(cudaFree(TensorPool[i][j]));
    
    TensorPool[i].clear();
  }
  TensorPool.clear();
  */
  
  for (auto& pair : NamedParamGrads)
  {
    std::cout << "Erasing gradient memory of: " << pair.first << "\n";
    cudaCheck(cudaFree(pair.second));
    NamedParamGrads.erase(pair.first);
  }

  NamedParamGrads.clear();


  //std::cout << "\n\nALL TENSORS ARE" << "\n\n";
  //for (auto& pair : NamedTensorsT)
  //  std::cout << pair.first << "\n"; 


  for (auto &pair: optimizer->NamedV)
    cudaCheck(cudaFree(pair.second));
    
  for (auto &pair: optimizer->NamedM)
    cudaCheck(cudaFree(pair.second));


  /*
  for (auto& pair : NamedTensorsT)
  {
    Tensor *tensor = pair.second;

    if (tensor->is_pinned)
    {
      std::cout << "Erasing tensor: " << pair.first << ", is pinned: " << tensor->is_pinned << "\n";
      tensor->Sync();
      cudaCheck(cudaFree(tensor->tensor_ptr));

      //delete[] tensor->cpu_tensor_ptr;
    }
  }
  */
  
  

  nn_mode = eval_mode;

  std::cout << "\n\n\n";
  return 0;
}

extern "C" float train()
{
  std::cout << "SETTING NN MODE TO TRAIN" << "\n\n";
  nn_mode = training_mode;
  return 0;
}




Value *BinaryTensorTensorExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  Value *LtensorName = Builder->CreateGlobalString(LHS->GetName());
  Value *RtensorName = Builder->CreateGlobalString(RHS->GetName());
  Value *object_name;


  
  // if is attribution
  if (Op == '=') {
  
    seen_var_attr=true;

    Value *RtensorPtr = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
    

    if (!LHS->GetIsVec())
    {
      VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
      LtensorName = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

      if (!LHSE)
        return LogErrorV("'=' left side expression must be a var.");
      
      //std::cout << "1 1 attr\n";


      Builder->CreateCall(TheModule->getFunction("AttrTensor"),
                          {LtensorName, RtensorPtr, scope_str, thread_id, has_grad});
      //std::cout << "Post attr call\n\n";
    } else
    {
      std::cout << "1 1 INDEXED attr\n";

      VecIdxExprAST *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
      LtensorName = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
      if (!LHSE)
        return LogErrorV("'=' left side expression must be a var.");

      if(LHSE->Idx[0]->GetType()!="tensor")
      {
        std::vector<Value *> idx_calc_args;
        idx_calc_args.push_back(LtensorName);
        for (int i=0; i<LHSE->Idx.size(); i++)
          idx_calc_args.push_back(LHSE->Idx[i]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad));
        Value *idx_at = Builder->CreateCall(TheModule->getFunction("CalculateIdxOffset"),
                              idx_calc_args);

        Builder->CreateCall(TheModule->getFunction("AttrTensorOnIdx"),
                            {LtensorName, RtensorPtr,
                             idx_at, thread_id});
      } else {
        VariableExprAST *idx = static_cast<VariableExprAST *>(LHSE->Idx[0].get());
        Value *idx_tensor_name = idx->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
        
        Builder->CreateCall(TheModule->getFunction("AttrTensorOnIdxTensor"), {LtensorName, idx_tensor_name, RtensorPtr, thread_id});

      }
    }

    seen_var_attr=false;
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  }




  std::string functionName = Builder->GetInsertBlock()->getParent()->getName().str();
  std::cout << "\nTensor Tensor for function: " << functionName << "\n";
  int forward_func = 0;
  if(ends_with(functionName, "forward"))
    forward_func = 1;
  forward_func = 1; // TODO: RemoveLastDim this line



  
  Value *LtensorPtr = LHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  Value *RtensorPtr = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);



  if (!LtensorPtr || !RtensorPtr)
    return nullptr;

  Function *CudaFn;

  std::cout << "Tensor tensor: " << LHS->GetName() << ", " << RHS->GetName() << "\n";
    


  Value *is_forward_func = ConstantInt::get(Type::getInt32Ty(*TheContext), forward_func);
  
  /*
  void *vec = &NamedDims[LHS->GetName()];
  Value* LLVMValue = ConstantInt::get(Type::getInt64Ty(*TheContext), reinterpret_cast<uint64_t>(vec));
  LLVMValue = Builder->CreateIntToPtr(LLVMValue, int8PtrTy);
  */

  
  Value *new_dims;

  switch (Op)
  {
  case '@':
    return Builder->CreateCall(TheModule->getFunction("CudaMult"),
                                    {is_forward_func,
                                     LtensorPtr, RtensorPtr, thread_id});
  case '/':
  {
    return Builder->CreateCall(TheModule->getFunction("CudaDiv"),
                                    {is_forward_func,
                                     LtensorPtr, RtensorPtr, thread_id});
  }
  case '+':
    return Builder->CreateCall(TheModule->getFunction("CudaAdd"),
                                    {is_forward_func,
                                     LtensorPtr, RtensorPtr, thread_id});
  case '*':
    return Builder->CreateCall(TheModule->getFunction("CudaHadamard"),
                                    {is_forward_func,
                                     LtensorPtr, RtensorPtr, thread_id});
  case '-':
    return Builder->CreateCall(TheModule->getFunction("CudaSub"),
                                    {is_forward_func,
                                     LtensorPtr, RtensorPtr, thread_id});
  case tok_equal:
    return Builder->CreateCall(TheModule->getFunction("CudaEqual"),
                               {is_forward_func, LtensorPtr, RtensorPtr, thread_id}, "cudaequal");
  case ':':
    return LtensorPtr;
  default:
    break;
  }
  

  std::string _error = "The operator " + ReverseToken(Op) + " is not implemented for operations between tensors";
  LogErrorS(_error);
  
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {LtensorName, RtensorName};
  return Builder->CreateCall(F, Ops, "binop");
}




Value *BinaryTensorPinnedExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  std::cout << "Binary Tensor Pinned codegen" << "\n";

  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  std::cout << "Binary Tensor Pinned codegen" << "\n";

  Value *LtensorName = Builder->CreateGlobalString(LHS->GetName());
  Value *object_name;



  // if is attribution
  if (Op == '=') {
  
    seen_var_attr=true;

    Value *RtensorPtr = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
    

    if (!LHS->GetIsVec())
    {
      VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
      LtensorName = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

      if (!LHSE)
        return LogErrorV("'=' left side expression must be a var.");
      std::cout << "1 2 attr\n";
      
      

      Builder->CreateCall(TheModule->getFunction("AttrTensorNoFree"),
                          {LtensorName, RtensorPtr, thread_id});
      std::cout << "Post attr call\n";
    } else
    {
      std::cout << "1 2 INDEXED attr\n";

      VecIdxExprAST *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
      LtensorName = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
      if (!LHSE)
        return LogErrorV("'=' left side expression must be a var.");


      std::vector<Value *> idx_calc_args;
      idx_calc_args.push_back(LtensorName);
      for (int i=0; i<LHSE->Idx.size(); i++)
        idx_calc_args.push_back(LHSE->Idx[i]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad));
      Value *idx_at = Builder->CreateCall(TheModule->getFunction("CalculateIdxOffset"),
                            idx_calc_args);

      
      Builder->CreateCall(TheModule->getFunction("AttrTensorOnIdx"),
                          {LtensorName, RtensorPtr,
                           idx_at, thread_id});
      
    }
    seen_var_attr=false;
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  }
  
}





Value *BinaryObjExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Value *LName = Builder->CreateGlobalString(LHS->GetName());
  Value *RName = Builder->CreateGlobalString(RHS->GetName());
  Value *object_name;

  


  // if is attribution
  if (Op == '=') {
  
    seen_var_attr=true;


    if (!LHS->GetIsVec())
    {
      std::cout << "\n\n3 3 attr\n";
      VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
      if (!LHSE)
        return LogErrorV("'=' object attribution destiny must be an object variable.");
      LName = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
      
      if (RHS->GetIsVec())
      {
        std::cout << "3 3 other INDEXED of RHS->GetIsVec() && RHS->GetType()==object" << "\n";
        VecIdxExprAST *RHSE = static_cast<VecIdxExprAST *>(RHS.get());
        RName = RHSE->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
        
        Builder->CreateCall(TheModule->getFunction("objAttr_var_from_vec"),
                                                        {LName, RName});
      } else {
        VariableExprAST *RHSE = static_cast<VariableExprAST *>(RHS.get());
        RName = RHSE->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
        
        Builder->CreateCall(TheModule->getFunction("objAttr_var_from_var"),
                                                        {LName, RName});

      }
    
    } else {
      std::cout << "\n\n3 3 other INDEXED attr\n";
      VecIdxExprAST *LHSE = static_cast<VecIdxExprAST *>(LHS.get());
      if (!LHSE)
        return LogErrorV("'=' object attribution destiny must be an object variable.");
      LName = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);


      std::cout << "ok" << "\n";
      
      if (RHS->GetIsVec())
      {
        std::cout << "3 3 other INDEXED of RHS->GetIsVec() && RHS->GetType()==object" << "\n";
        VecIdxExprAST *RHSE = static_cast<VecIdxExprAST *>(RHS.get());
        RName = RHSE->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
        
        Builder->CreateCall(TheModule->getFunction("objAttr_vec_from_vec"),
                                                        {LName, RName});
      } else {
        std::cout << "3 3 VEC FROM VAR" << "\n";
        VariableExprAST *RHSE = static_cast<VariableExprAST *>(RHS.get());
        RName = RHSE->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
        
        Builder->CreateCall(TheModule->getFunction("objAttr_vec_from_var"),
                                                        {LName, RName});

      }


    }
    seen_var_attr=false;
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  }
  
}




Value *ConcatStringsExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Special case '=' because we don't want to emit the LHS as an expression.

  

  if (Op == '=') {

    //std::cout << "\n0 0 ATTRIBUTION" << "\n\n\n";

    seen_var_attr=true;
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
    Value *Lvar_name = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);


    NameSolverAST *name_solver = static_cast<NameSolverAST *>(LHSE->NameSolver.get());
    std::string Lname = std::get<0>(name_solver->Names[0]);
    std::string LType = LHS->GetType();


    if (!LHSE)
      return LogErrorV("'=' destiny must be a variable.");
    // Codegen the RHS.
    
    Value *Val = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

    if (!Val)
    {
      seen_var_attr=false;
      return nullptr;
    }

    // Look up the name.
    if (LType=="float") {
      Builder->CreateCall(TheModule->getFunction("StoreOnDemand"),
                                                  {Lvar_name,
                                                   Val});

    } else if (LType=="str") {


      Builder->CreateCall(TheModule->getFunction("StoreStrOnDemand"),
                                                  {Lvar_name,
                                                   Val});
                                                   

    } else if (LType=="str_vec") {

      //std::cout << "ATTRIBUTING TO STRING VEC: " << Lname << "\n";
      Value *Variable = NamedStrVecs[Lname];
      
      if(LHS->GetSelf())
        Builder->CreateCall(TheModule->getFunction("StoreStrVecOnDemand"),
                                                  {Lvar_name,
                                                   Val});
      else
        Builder->CreateStore(Val, Variable);

    } else if (LType=="float_vec") {

      //std::cout << "ATTRIBUTING TO FLOAT VEC: " << Lname << ", type: " << Type << ", is vec: " << LHS->GetIsVec() << "\n";

      Value *Variable = NamedFloatVecs[Lname];
      

      if(LHS->GetSelf())
      {
        if(LHS->GetIsVec())
        {
          VecIdxExprAST *LHSV = static_cast<VecIdxExprAST *>(LHS.get());
          

          Builder->CreateCall(TheModule->getFunction("StoreFloatVecOnDemandOnIdx"),
                                                  {Lvar_name,
                                                   LHSV->Idx[0]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad),
                                                   Val});

        } else
          Builder->CreateCall(TheModule->getFunction("StoreFloatVecOnDemand"),
                                                  {Lvar_name,
                                                   Val});
      }
      else
      {
        if(LHS->GetIsVec())
        {
          // TODO: Implement non-object-attr float vector index attribution.
        } else
          Builder->CreateStore(Val, Variable);
      }
        

    } else {
      
      seen_var_attr=false;
      
      
      Builder->CreateCall(TheModule->getFunction("StoreOnDemand"),
                                                  {Lvar_name,
                                                   Val});
      

      //std::string _error = "Could not find variable " + Lname + ".";
      //return LogErrorV(_error);
    }

    seen_var_attr=false;
    return Val;
  }


  

  Value *L = LHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  Value *R = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  
  if (!L || !R)
    return nullptr;


    switch (Op) {
    case '+':
      return Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                          {L, R});
    default:
      LogErrorS("The only string operations supported are '+' and '='.");
      break;
    }
  

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {L, R};
  return Builder->CreateCall(F, Ops, "binop");
}








Value *BinaryExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Special case '=' because we don't want to emit the LHS as an expression.
  if (Op == '=') {

    //std::cout << "\n0 0 ATTRIBUTION" << "\n\n\n";

    seen_var_attr=true;
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
    Value *Lvar_name = LHSE->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);


    NameSolverAST *name_solver = static_cast<NameSolverAST *>(LHSE->NameSolver.get());
    std::string Lname = std::get<0>(name_solver->Names[0]);
    std::string LType = LHS->GetType();


    if (!LHSE)
      return LogErrorV("'=' destiny must be a variable.");
    // Codegen the RHS.
    
    Value *Val = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

    if (!Val)
    {
      seen_var_attr=false;
      return nullptr;
    }

    // Look up the name.
    if (LType=="float") {
      Builder->CreateCall(TheModule->getFunction("StoreOnDemand"),
                                                  {Lvar_name,
                                                   Val});

    } else if (LType=="str") {


      Builder->CreateCall(TheModule->getFunction("StoreStrOnDemand"),
                                                  {Lvar_name,
                                                   Val});
                                                   

    } else if (LType=="str_vec") {

      //std::cout << "ATTRIBUTING TO STRING VEC: " << Lname << "\n";
      Value *Variable = NamedStrVecs[Lname];
      
      if(LHS->GetSelf())
        Builder->CreateCall(TheModule->getFunction("StoreStrVecOnDemand"),
                                                  {Lvar_name,
                                                   Val});
      else
        Builder->CreateStore(Val, Variable);

    } else if (LType=="float_vec") {

      //std::cout << "ATTRIBUTING TO FLOAT VEC: " << Lname << ", type: " << Type << ", is vec: " << LHS->GetIsVec() << "\n";

      Value *Variable = NamedFloatVecs[Lname];
      

      if(LHS->GetSelf())
      {
        if(LHS->GetIsVec())
        {
          VecIdxExprAST *LHSV = static_cast<VecIdxExprAST *>(LHS.get());
          

          Builder->CreateCall(TheModule->getFunction("StoreFloatVecOnDemandOnIdx"),
                                                  {Lvar_name,
                                                   LHSV->Idx[0]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad),
                                                   Val});

        } else
          Builder->CreateCall(TheModule->getFunction("StoreFloatVecOnDemand"),
                                                  {Lvar_name,
                                                   Val});
      }
      else
      {
        if(LHS->GetIsVec())
        {
          // TODO: Implement non-object-attr float vector index attribution.
        } else
          Builder->CreateStore(Val, Variable);
      }
        

    } else {
      
      seen_var_attr=false;
      
      
      Builder->CreateCall(TheModule->getFunction("StoreOnDemand"),
                                                  {Lvar_name,
                                                   Val});
      

      //std::string _error = "Could not find variable " + Lname + ".";
      //return LogErrorV(_error);
    }

    seen_var_attr=false;
    return Val;
  }


  

  Value *L = LHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  Value *R = RHS->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  
  if (!L || !R)
    return nullptr;


    switch (Op) {
    case '+':
      return Builder->CreateFAdd(L, R, "addtmp");
    case ':':
      return L;
    case tok_space:
      return R;
    case '-':
      return Builder->CreateFSub(L, R, "subtmp");
    case '*':
      return Builder->CreateFMul(L, R, "multmp");
    case '/':
      return Builder->CreateFDiv(L, R, "divtmp");
    case '%':
      return Builder->CreateFRem(L, R, "remtmp");
    case 77:
      return LogErrorV("GOTCHA");
    case '<':
      L = Builder->CreateFCmpULT(L, R, "cmptmp");
      // Convert bool 0/1 to float 0.0 or 1.0
      return Builder->CreateUIToFP(L, Type::getFloatTy(*TheContext), "booltmp");
    case '>':
      L = Builder->CreateFCmpULT(R, L, "cmptmp");
      // Convert bool 0/1 to float 0.0 or 1.0
      return Builder->CreateUIToFP(L, Type::getFloatTy(*TheContext), "booltmp");
    case tok_equal:
      L = Builder->CreateFCmpUEQ(L, R, "cmptmp");
      // Convert bool 0/1 to float 0.0 or 1.0
      return Builder->CreateUIToFP(L, Type::getFloatTy(*TheContext), "booltmp");
    case tok_diff:
      L = Builder->CreateFCmpUNE(L, R, "cmptmp");
      // Convert bool 0/1 to float 0.0 or 1.0
      return Builder->CreateUIToFP(L, Type::getFloatTy(*TheContext), "booltmp");
    default:
      break;
    }
  

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "Operator not found.");

  Value *Ops[] = {L, R};
  return Builder->CreateCall(F, Ops, "binop");
}






Value *UnaryExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  Value *OperandV = Operand->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  if (!OperandV)
    return nullptr;
  
  
  
  //std::cout << "Operand type: " << Operand->GetType();
  if (Opcode=='-')
  {
    //std::cout << "\n\n\n\n\n\nIT'S A MINUS " << Operand->GetType() << "\n\n\n\n\n\n\n";
    if (Operand->GetType()=="tensor")
    {
      Value *tensor_name = Builder->CreateGlobalString(Operand->GetName());

      std::string pre_dot = Operand->GetPreDot();
      bool is_self = Operand->GetSelf();
      bool is_attr = Operand->GetIsAttribute();

      if (is_attr) { // Gets from pre_dot if it is a class attribute
        Value * object_name = Builder->CreateGlobalString(pre_dot);

        tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                          {object_name, tensor_name});
      }
      if (is_self)
        tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                          {first_arg, tensor_name});
      if (!(is_self||is_attr))
        tensor_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                {scope_str, tensor_name});
        

      Value *tensorPtr = Builder->CreateCall(TheModule->getFunction("LoadTensor"),
                                              {tensor_name});
      Value *R = ConstantFP::get(Type::getFloatTy(*TheContext), -1);

      return Builder->CreateCall(TheModule->getFunction("CudaScalarMult"),
                                {tensorPtr, R, thread_id}, "cudascalarmult");
    }
    return Builder->CreateFMul(ConstantFP::get(Type::getFloatTy(*TheContext), -1),
                              OperandV, "multmp");
  }

  //std::cout << "Opcode: " << Opcode << "\n";


  if (Opcode='!')
  {
    return Builder->CreateCall(TheModule->getFunction("logical_not"), {OperandV});
  }
  if (Opcode=';')
    return ConstantFP::get(Type::getFloatTy(*TheContext), 0);
  

  Function *F = getFunction(std::string("unary") + Opcode);
  if (!F)
    return LogErrorV("Operador unário desconhecido.");

  return Builder->CreateCall(F, OperandV, "unop");
}


Value *IfExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Value *CondV = Cond->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  if (!CondV)
    return nullptr;

  // Convert condition to a bool by comparing equal to 0.0.
  CondV = Builder->CreateFCmpONE(
      CondV, ConstantFP::get(*TheContext, APFloat(0.0)), "ifcond");

  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  BasicBlock *ThenBB  = BasicBlock::Create(*TheContext, "then", TheFunction);
  BasicBlock *ElseBB  = BasicBlock::Create(*TheContext, "else");
  BasicBlock *MergeBB = BasicBlock::Create(*TheContext, "ifcont");

  Builder->CreateCondBr(CondV, ThenBB, ElseBB);

  // Emit then value.
  Builder->SetInsertPoint(ThenBB);

  
  Value *ThenV;
  for (auto &then_body : Then)
    ThenV = then_body->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  

  if (!ThenV)
    return nullptr;


  Builder->CreateBr(MergeBB);
  // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
  ThenBB = Builder->GetInsertBlock();

  // Emit else block.
  TheFunction->insert(TheFunction->end(), ElseBB);
  Builder->SetInsertPoint(ElseBB);


  Value *ElseV;
  for (auto &else_body : Else)
    ElseV = else_body->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

  if (!ElseV)
    return nullptr;

    

  Builder->CreateBr(MergeBB);
  // Codegen of 'Else' can change the current block, update ElseBB for the PHI.
  ElseBB = Builder->GetInsertBlock();

  // Emit merge block.
  TheFunction->insert(TheFunction->end(), MergeBB);
  Builder->SetInsertPoint(MergeBB);
  PHINode *PN = Builder->CreatePHI(Type::getFloatTy(*TheContext), 2, "iftmp");

  PN->addIncoming(ThenV, ThenBB);
  PN->addIncoming(ElseV, ElseBB);
  
  return PN;
}

// Output for-loop as:
//   var = alloca float
//   ...
//   start = startexpr
//   store start -> var
//   goto loop
// loop:
//   ...
//   bodyexpr
//   ...
// loopend:
//   step = stepexpr
//   endcond = endexpr
//
//   curvar = load var
//   nextvar = curvar + step
//   store nextvar -> var
//   br endcond, loop, endloop
// outloop:

Value *ForExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Create an alloca for the variable in the entry block.
  AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);

  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  if (!StartVal)
    return nullptr;

  Value *_zero = ConstantFP::get(*TheContext, APFloat(0.0));



  Value *var_name = Builder->CreateGlobalString(VarName);
  var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                    {scope_str, var_name});

  Builder->CreateCall(TheModule->getFunction("StoreOnDemandNoFree"),
                                                  {var_name, StartVal});

  // Store the value into the alloca.
  //Builder->CreateStore(StartVal, Alloca);

  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock *CondBB = BasicBlock::Create(*TheContext, "cond", TheFunction);
  BasicBlock *LoopBB  = BasicBlock::Create(*TheContext, "loop");
  BasicBlock *AfterBB  = BasicBlock::Create(*TheContext, "after");



  // Insert an explicit fall through from the current block to the LoopBB.
  Builder->CreateBr(CondBB);

  
  Builder->SetInsertPoint(CondBB);

  // Within the loop, the variable is defined equal to the PHI node.  If it
  // shadows an existing variable, we have to restore it outside this scope
  //Value *OldVal = NamedValues[VarName];
  //NamedValues[VarName] = Alloca;



  // Emit the body of the loop.  This, like any other expr, can change the
  // current BB.  Note that we ignore the value computed by the body, but don't
  // allow an error.
  
  

  // Emit the step value.
  Value *StepVal = nullptr;
  if (Step) {
    StepVal = Step->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
    if (!StepVal)
      return nullptr;
  } 


  // Compute the end condition.
  Value *EndCond = End->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  if (!EndCond)
    return nullptr;

  // Convert condition to a bool by comparing equal to 0.0.
  EndCond = Builder->CreateFCmpONE(
      EndCond, _zero, "loopcond");




  // conditional goto branch
  Builder->CreateCondBr(EndCond, LoopBB, AfterBB);




  // codegen body and increment
  TheFunction->insert(TheFunction->end(), LoopBB);
  Builder->SetInsertPoint(LoopBB);

  int j=0;
  for (auto &body : Body)
    body->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  Value *CurVar = Builder->CreateCall(TheModule->getFunction("LoadOnDemandNoFree"), {var_name});
  Value *NextVar = Builder->CreateFAdd(CurVar, StepVal, "nextvar"); // Increment
  Builder->CreateCall(TheModule->getFunction("StoreOnDemandNoFree"),
                                                  {var_name, NextVar});

  
  
  Builder->CreateBr(CondBB);




  // when the loop body is done, return the insertion point to outside the for loop
  TheFunction->insert(TheFunction->end(), AfterBB);
  Builder->SetInsertPoint(AfterBB);

  Builder->CreateCall(TheModule->getFunction("FreeChar"), {var_name});
  // Restore the unshadowed variable.
  //if (OldVal)
  //  NamedValues[VarName] = OldVal;
  //else
  //  NamedValues.erase(VarName);

  // for expr always returns 0.0.
  return Constant::getNullValue(Type::getFloatTy(*TheContext));
}



Value *WhileExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  
  Function* TheFunction = Builder->GetInsertBlock()->getParent();

  // Create blocks for loop condition, loop body, and after loop
  BasicBlock *CondBB = BasicBlock::Create(*TheContext, "cond_while", TheFunction);
  BasicBlock *LoopBB = BasicBlock::Create(*TheContext, "loop_while", TheFunction);
  BasicBlock *AfterBB = BasicBlock::Create(*TheContext, "end_while", TheFunction);

  // Jump to the condition block
  Builder->CreateBr(CondBB);

  // Insert the condition check block
  Builder->SetInsertPoint(CondBB);

  // Generate the condition code
  Value* condVal = Cond->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  if (!condVal)
    return nullptr;

  Value *_zero = ConstantFP::get(*TheContext, APFloat(0.0));
  condVal = Builder->CreateFCmpONE(condVal, _zero, "loopcond");

  // Create the conditional branch
  Builder->CreateCondBr(condVal, LoopBB, AfterBB);

  // Insert the loop body block
  Builder->SetInsertPoint(LoopBB);

  // Generate the loop body code
  for (auto &body : Body)
    body->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

  // After the loop body, go back to the condition check
  Builder->CreateBr(CondBB);

  // Insert the after loop block
  Builder->SetInsertPoint(AfterBB);

  return Constant::getNullValue(Type::getFloatTy(*TheContext));
}



Function *codegenAsyncFunction(std::vector<std::unique_ptr<ExprAST>> &asyncBody, Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  

  // find existing unique function name (_async_1, _async_2, _async_3 etc)
  int fnIndex = 1;
  while (TheModule->getFunction("__async_" + std::to_string(fnIndex)))
    fnIndex++;
  
  CudaStreams *thread_stream = AllocateStream();
  ThreadsStream[fnIndex] = thread_stream->stream;

  // Create function for this async function
  llvm::Type *int8PtrTy = Type::getInt8Ty(*TheContext)->getPointerTo();

  FunctionType *asyncFunTy = FunctionType::get(
                                            int8PtrTy,
                                            {int8PtrTy},
                                            false);
                                            
  std::string functionName = "__async_" + std::to_string(fnIndex);
  Function *asyncFun =
      Function::Create(asyncFunTy,
                             Function::ExternalLinkage,
                             functionName,
                             TheModule.get());


  
  // emit EntryBB value
  std::cout << "\n\nfunction * get basic block for function: " << functionName << "\n";
  BasicBlock *BB = BasicBlock::Create(*TheContext, "async_bb", asyncFun);
  Builder->SetInsertPoint(BB);
  

  // define body of function
  Value *V;

  for (auto &body : asyncBody)
    V = body->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);



  if (V)
  {
    
    std::cout << "create return" << "\n";
    Builder->CreateRet(Constant::getNullValue(int8PtrTy));
    

    std::string functionError;
    llvm::raw_string_ostream functionErrorStream(functionError);

    if (verifyFunction(*asyncFun, &functionErrorStream)) {
      functionErrorStream.flush();
      llvm::errs() << "Function verification failed:\n" << functionError << "\n";
    } 

    verifyModule(*TheModule);
    return asyncFun;
  }
  
  std::cout << "ERASING ASYNC FROM PARENT" << "\n";
  asyncFun->eraseFromParent();

  return nullptr;
}


//int pthread_create(pthread_t *thread, pthread_attr_t *attr,
//                   void *(*start_routine) (void *arg), void *arg);



extern "C" void pthread_create_aux(pthread_t *thread, pthread_attr_t *attr,
                   void *(*start_routine) (void *arg), void *arg)
{
  std::cout << "Creating thread" << "\n";
  pthread_create(thread, attr, start_routine, arg);
  std::cout << "Created" << "\n";
}


extern "C" void pthread_join_aux(pthread_t thread)
{
  std::cout << "Joining " << thread <<  "\n";
  void **value_ptr;
  value_ptr = nullptr;

  pthread_join(thread, value_ptr);
  std::cout << "Joined: " << thread << "\n";
}



std::vector<Value *> thread_pointers;


Value *AsyncExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  
  // Create/Spawn Threads

  BasicBlock *CurrentBB = Builder->GetInsertBlock();


  
  
  //std::cout << "\nAsync get insert block for function: " << functionName << "\n\n";


  Function *asyncFun = codegenAsyncFunction(std::ref(Body), first_arg, scope_str, previous_scope, thread_id, has_grad);


  Builder->SetInsertPoint(CurrentBB);

  
  Function *pthread_create = TheModule->getFunction("pthread_create_aux");


  PointerType *pthreadTy = Type::getInt8Ty(*GlobalContext)->getPointerTo();
  Value *pthreadPtr = Builder->CreateAlloca(pthreadTy, nullptr);
  
  

  Value *voidPtrNull = Constant::getNullValue(
      Type::getInt8Ty(*TheContext)->getPointerTo());


  
  Builder->CreateCall(pthread_create,
    {pthreadPtr,
     voidPtrNull,
     asyncFun,
     voidPtrNull}
  );
  
  std::cout << "Created join call" << "\n";



  thread_pointers.push_back(pthreadPtr);

  return pthreadPtr;
}



Value *FinishExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();




  //std::cout << "\n\nFinish codegen for: " << functionName <<  "\n";


  

  for (int i=0; i < Bodies.size(); i++)
    Bodies[i]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
  

  
  

  PointerType *pthreadTy = Type::getInt8Ty(*GlobalContext)->getPointerTo();

  Function *pthread_join = TheModule->getFunction("pthread_join_aux");


  //std::cout << "\n\n\n\nFINISH HAS " << thread_pointers.size() << " ASYNC EXPRESSIONS "  << "\n\n\n\n\n";


  for (Value *pthreadPtr : thread_pointers)
  {
    Value *pthread = Builder->CreateLoad(pthreadTy, pthreadPtr);

    Builder->CreateCall(pthread_join,
                        {pthread});
    
  }
  
  thread_pointers.clear();
  
  return ConstantFP::get(*TheContext, APFloat(0.0f));
}


Value *LockExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad){
  
  Builder->CreateCall(TheModule->getFunction("LockMutex"), {Builder->CreateGlobalString(Name)});

  for (auto &body : Bodies)
    body->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

  Builder->CreateCall(TheModule->getFunction("UnlockMutex"), {Builder->CreateGlobalString(Name)});

  return ConstantFP::get(*TheContext, APFloat(0.0f));
}


Value *NoGradExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad){
  
  has_grad  = ConstantInt::get(Type::getInt32Ty(*TheContext), 0);
  for (auto &body : Bodies)
    body->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

  
  return ConstantFP::get(*TheContext, APFloat(0.0f));
}





Value *ReturnExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {

  for (int i=0; i<Destiny.size(); i++)
  {
    //TODO: add self and attr to return
    
    std::string name, type, l_name, l_type;
    bool is_vec, l_is_vec;

    name   = Destiny[i]->GetName();
    type   = Destiny[i]->GetType();
    is_vec = Destiny[i]->GetIsVec();

    Value *_name = Builder->CreateGlobalString(name);

    std::cout << "\nRETURNING: " << name << ", type: " << type << ", is vec: " << is_vec <<  "\n\n";

    if (!IsAs[i])
    {
      if(type=="tensor")
      {
        VariableExprAST *destiny = static_cast<VariableExprAST *>(Destiny[i].get());
        destiny->NameSolver->SetSolverIncludeScope(false);
        _name = destiny->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

        Builder->CreateCall(TheModule->getFunction("RemoveTensorScope"),
                                            {_name, scope_str,
                                             _name, previous_scope,
                                             thread_id});
      }
    } else {
      l_name   = Vars[i]->GetName();
      l_type   = Vars[i]->GetType();
      l_is_vec = Vars[i]->GetIsVec();

      std::cout << "l_name: " << l_name << " l_type: " << l_type << ", l_is_vec: " << l_is_vec << "\n";

      if (!is_vec)
      {
        VariableExprAST *destiny = static_cast<VariableExprAST *>(Destiny[i].get());
        destiny->NameSolver->SetSolverIncludeScope(false);
        _name = destiny->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

        
        VariableExprAST *var = static_cast<VariableExprAST *>(Vars[i].get());
        var->NameSolver->SetSolverIncludeScope(false);
        Value *_l_name = var->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

        if (l_type=="tensor"||type=="tensor")
        {
          Builder->CreateCall(TheModule->getFunction("RemoveTensorScope"),
                                              {_l_name, scope_str,
                                               _name,   previous_scope,
                                               thread_id});
        }
      } else {

        VecIdxExprAST *destiny = static_cast<VecIdxExprAST *>(Destiny[i].get());
        if (!destiny)
          return LogErrorV("Could not deal with return expression");
        destiny->NameSolver->SetSolverIncludeScope(false);
        _name = destiny->NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
        

        std::vector<Value *> idx_calc_args;
        idx_calc_args.push_back(Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                      {previous_scope, _name}));
        for (int i=0; i<destiny->Idx.size(); i++)
          idx_calc_args.push_back(destiny->Idx[i]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad));
        Value *idx_at = Builder->CreateCall(TheModule->getFunction("CalculateIdxOffset"),
                              idx_calc_args);

        
        Value *_l_name = Builder->CreateGlobalString(l_name);
        Builder->CreateCall(TheModule->getFunction("RemoveTensorScopeAttrOnIndex"),
                                              {_l_name, scope_str,
                                               _name, previous_scope,
                                               idx_at, thread_id});
      }
    }
  }

  return ConstantFP::get(*TheContext, APFloat(0.0));
}




// Create Float Var
Value *VarExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  std::vector<Value *> OldBindings;

  Function *TheFunction = Builder->GetInsertBlock()->getParent();


  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();

    // Emit the initializer before adding the variable to scope, this prevents
    // the initializer from referencing the variable itself, and permits stuff
    // like this:
    //  var a = 1 in
    Value *InitVal;
    if (Init) {
      InitVal = Init->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
      if (!InitVal)
        return nullptr;
    } else { // If not specified, use 0.0.
      InitVal = ConstantFP::get(*TheContext, APFloat(0.0));
    }



    Value *var_name = Builder->CreateGlobalString(VarName);
    var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                        {scope_str, var_name});


    Builder->CreateCall(TheModule->getFunction("AddFloatToScopeCleanList"),
                                        {scope_str, var_name});

    Builder->CreateCall(TheModule->getFunction("StoreOnDemand"),
                                        {var_name, InitVal});
                                                  
    
  }


  return ConstantFP::get(*TheContext, APFloat(0.0));
}





Value *StrExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  //std::vector<AllocaInst *> OldBindings;

  Function *TheFunction = Builder->GetInsertBlock()->getParent();


  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();

    // Emit the initializer before adding the variable to scope, this prevents
    // the initializer from referencing the variable itself, and permits stuff
    // like this:
    //  var a = 1 in
    Value *InitVal;
    if (Init) {
      InitVal = Init->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
      if (!InitVal)
        return nullptr;
    } else { // If not specified, use 0.0.
      InitVal = Builder->CreateGlobalString("");
    }



      
    // Remember the old variable binding so that we can restore the binding when
    // we unrecurse.
    
    Value *var_name, *scopeless_name;
    var_name = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                            {Builder->CreateGlobalString(VarName)});
    

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeRight"),
                                            {first_arg, var_name});
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeRight"),
                                            {scope_str, var_name});
    

    
    Builder->CreateCall(TheModule->getFunction("AddToScopeCleanList"),
                        {scope_str,
                         Builder->CreateCall(TheModule->getFunction("CopyString"), {var_name})
                        });
    
        

                        
    Builder->CreateCall(TheModule->getFunction("StoreStrOnDemand"),
                                                  {var_name,
                                                   InitVal});
  }

  
  return ConstantFP::get(*TheContext, APFloat(0.0f));
}


Value *StrVecExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  //std::vector<AllocaInst *> OldBindings;

  Function *TheFunction = Builder->GetInsertBlock()->getParent();


  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();

    // Emit the initializer before adding the variable to scope, this prevents
    // the initializer from referencing the variable itself, and permits stuff
    // like this:
    //  var a = 1 in
    Value *InitVal;
    if (Init) {
      InitVal = Init->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
      if (!InitVal)
        return nullptr;
    } else { // If not specified, use 0.0.
      InitVal = ConstantFP::get(*TheContext, APFloat(0.0));
    }


    AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);
    Builder->CreateStore(InitVal, Alloca);
      
    // Remember the old variable binding so that we can restore the binding when
    // we unrecurse.
    //std::cout << "STRING CODEGEN FOR " << VarName << "\n";
    //OldBindings.push_back(NamedStrVecs[VarName]);

    
    // Remember this binding.

    if (Type=="str")
      NamedStrVecs[VarName] = Alloca;
    if (Type=="float")
      NamedFloatVecs[VarName] = Alloca;
    
    
  }


  //for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
  //  NamedStrVecs[VarNames[i].first] = OldBindings[i];

  
  return ConstantFP::get(*TheContext, APFloat(0.0f));
}


extern "C" float InitObjectVecWithNull(char *name, float vec_size) 
{
  std::cout << "InitObjectVecWithNull of " << name << " with vec_size " << vec_size << "\n\n\n\n";

  for (int i=0; i<vec_size; i++)
  {
    std::string indexed_name = name + std::to_string(i);
    objectVecs[indexed_name] = "nullptr";
  }
  
  delete[] name; //TODO: Break?
  return 0;
}

extern "C" float is_null(char *name)
{
  //std::cout << "\n\nIS NULL OF: " << name << "\n\n\n";

  if (objectVecs[name]=="nullptr")
    return 1;
  return 0;
}

Value *NewVecExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));

  std::vector<Value *> values;

  values.push_back(thread_id);
  for (int i=0; i<Values.size(); i++)
    values.push_back(Values[i]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad));



  return Builder->CreateCall(TheModule->getFunction("NewVecToTensor"), values);
}


Value *ObjectExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  Value *init;
  if (Init)
    init = Init->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);

  // Register all variables and emit their initializer.

  for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
  {
    const std::string &VarName = VarNames[i].first;

    Value *var_name;// = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});
    
    std::string pre_dot = GetPreDot();
    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();
    
    if (!GetIsVec())
    {
      var_name = Builder->CreateGlobalString(VarName);

      if (is_self||is_attr) 
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                              {first_arg, var_name});
      Builder->CreateCall(TheModule->getFunction("InstantiateObject"),
                                              {scope_str, var_name});
    }
    else if (Init) // init of vec[size]
    {
      //var_name = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});
      var_name = Builder->CreateGlobalString(VarName);

      if (is_self||is_attr) 
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"), //TODO: Break?
                                              {first_arg, var_name});
      if (!(is_self||is_attr))
        var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"), //TODO: Break?
                                              {scope_str, var_name});

      //Value *object_hash = Builder->CreateCall(TheModule->getFunction("RandomStrOnDemand"), {});
      //var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
      //                                        {object_hash, var_name});


      Builder->CreateCall(TheModule->getFunction("InitObjectVecWithNull"),
                                                {var_name, init});
    } else
    {}
  }


  return ConstantFP::get(*TheContext, APFloat(0.0));
}




extern "C" void *randu_like(int thread_id, Tensor tensor)
{
  float dims_prod = tensor.dims_prod;

  float *tensor_ptr, *tensor_cpu;

  tensor_cpu = make_random_float_uniform(dims_prod);

  cudaStream_t stream = ThreadsStream[thread_id];
  cudaMalloc(&tensor_ptr, dims_prod*sizeof(float));
  cudaMemcpyAsync(tensor_ptr, tensor_cpu, dims_prod*sizeof(float), cudaMemcpyHostToDevice, stream);
  delete[] tensor_cpu;

  Tensor *new_tensor = createTensor(tensor_ptr, tensor.dims, dims_prod, false, "");
  new_tensor->op = randu_like_op;
  return new_tensor;
}





std::vector<float> cur_dim;

extern "C" float StoreDimsOnDemand(char *tensor_name, float d)
{
  std::vector<float> dims;
  
  if (NamedDims.count(tensor_name)>0)
    dims = NamedDims[tensor_name];

  dims.push_back(d);

  NamedDims[tensor_name] = dims;
  return 0;
}




extern "C" float print_randoms(float N, float std) {
    //float std = sqrt(2/fan_in);
    
    int n = (int) N;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, std);

    float* arr = (float*)malloc(n * sizeof(float));
    std::cout << "[";
    for (size_t i = 0; i < n; i++)
      std::cout << dist(gen) << " ";
    std::cout << "]";

    return 0;
}



extern "C" float CreateTensorOnDemand(char *tensor_name, char *scopeless_name, char *init, int is_weight, int thread_id, char *scope)
{
  //std::cout << "CREATING TENSOR " << tensor_name << " AT THREAD: " << thread_id << "\n";

  Tensor *tensor;

  std::vector<float> dims = NamedDims[tensor_name];
  NamedDims[tensor_name].clear(); //TODO: Global vars are bad with threads.

  int product = DimsProd(dims);

  float *tensor_ptr;
  float *tensor_cpu;

  if(product>0)
  {
    if (std::strcmp(init, "randu") == 0)
      tensor_cpu = make_random_float_uniform(product);
    if (std::strcmp(init, "zeros") == 0)
      tensor_cpu = make_zeros_float(product);
    if (std::strcmp(init, "ones") == 0)
      tensor_cpu = make_ones_float(product);
    if (std::strcmp(init, "normal") == 0)
      tensor_cpu = make_normal(product);
    if (std::strcmp(init, "xavu") == 0)
      tensor_cpu = make_xavier_uniform_float(product, dims[dims.size()-1], dims[dims.size()-2]);
    if (std::strcmp(init, "xavu_relu") == 0)
      tensor_cpu = make_xavier_uniform_float_relu(product, dims[dims.size()-1], dims[dims.size()-2]);
    if (std::strcmp(init, "xavu_tanh") == 0)
      tensor_cpu = make_xavier_uniform_float_tanh(product, dims[dims.size()-1], dims[dims.size()-2]);
    if (std::strcmp(init, "he_normal_relu") == 0)
      tensor_cpu = make_he_normal_float_relu(product, dims[dims.size()-1]);
    if (std::strcmp(init, "init_gpt") == 0)
      tensor_cpu = make_gpt_init(product);
    if (std::strcmp(init, "int") == 0)
      tensor_cpu = make_random_int(product, 10);
    if (std::strcmp(init, "arange") == 0)
      tensor_cpu = make_arange(product);
    if (std::strcmp(init, "binary") == 0)
      tensor_cpu = make_random_int(product, 1);

    cudaCheck(cudaGetLastError());
    std::string _name = "create tensor ";
    _name = _name + tensor_name;
    tensor_ptr = get_from_pool(thread_id, product, _name);
    //std::cout << "cpy of: " << tensor_name << "\n";

    cudaStream_t stream = ThreadsStream[thread_id];
    cudaCheck(cudaMemcpyAsync(tensor_ptr, tensor_cpu, product*sizeof(float), cudaMemcpyHostToDevice, stream));
    //cudaStreamSynchronize(stream);
    delete[] tensor_cpu;
  }


  
  /*
  if(NamedTensorsT.count(tensor_name)>0)
  {
    tensor = NamedTensorsT[tensor_name];
    if (tensor!=nullptr)
    
      delete tensor;
      //cudaCheck(cudaFree(aux_ptr));
  }
  */
  
  


  tensor = createTensor(tensor_ptr, dims, product, true, tensor_name);
  tensor->scopeless_name = scopeless_name;
  if((bool)is_weight)
    tensor->SetIsWeight();
  tensor->op = create_tensor_op;

  
  NamedTensorsT[tensor_name] = tensor;

  
  delete[] tensor_name;
  delete[] scopeless_name;


  return 0;
}


Value *TensorExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    
    
    Value *var_name, *scopeless_name, *init;
    
    init = Builder->CreateGlobalString(TensorInit);
    var_name = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                            {Builder->CreateGlobalString(VarName)});

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeRight"),
                                            {first_arg, var_name});
    scopeless_name = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                            {var_name});
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStrFreeRight"),
                                            {scope_str, var_name});

    Value *aux;
    for (int j=0; j<V_Dims.size(); j++)
    {
      aux = V_Dims[j]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
      Builder->CreateCall(TheModule->getFunction("StoreDimsOnDemand"),
                                                  {var_name, aux});
    }

    Builder->CreateCall(TheModule->getFunction("CreateTensorOnDemand"),
                                              {var_name, scopeless_name, init,
                                               ConstantInt::get(Type::getInt32Ty(*TheContext), IsWeight), thread_id,
                                               scope_str});
  }


  return ConstantFP::get(*TheContext, APFloat(0.0));
}



extern "C" void CreatePinnedTensorOnDemand(char *tensor_name, char *init)
{
  std::vector<float> dims = NamedDims[tensor_name];
  NamedDims[tensor_name].clear();
  Tensor *tensor;

  int product = DimsProd(dims);
  float *tensor_ptr, *pool_tensor;
  float *tensor_cpu;


  cudaMallocHost(&tensor_cpu, product*sizeof(float));
  //tensor_cpu = new float[product];

  for (int i = 0; i < product; ++i) {
    tensor_cpu[i] = 0.0f;
  }
  

  cudaMalloc(&tensor_ptr, product*sizeof(float));  
  tensor = createPinned(tensor_ptr, tensor_cpu, dims, product, tensor_name);
  NamedTensorsT[tensor_name] = tensor;
  

  
  // pinned tensors are 1 pool tensor behind.
  std::vector<float> pool_dims = dims;
  pool_dims.erase(pool_dims.begin());
  float pool_product = DimsProd(pool_dims);
  pool_tensor = get_from_pool(0, pool_product, "create pinned");
  move_to_pool(0, pool_product, pool_tensor, "create pinned");
  
}

Value *PinnedTensorExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));



  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();

    // Emit the initializer before adding the variable to scope, this prevents
    // the initializer from referencing the variable itself, and permits stuff
    // like this:
    //  var a = 1 in
    Value *InitVal;
    if (Init) {
      InitVal = Init->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
      if (!InitVal)
        return nullptr;
    } else { // If not specified, use 0.0.
      InitVal = ConstantFP::get(*TheContext, APFloat(0.0));
    }


   Value *var_name = Builder->CreateGlobalString(VarName);

    std::string pre_dot = GetPreDot();
    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {first_arg, var_name});
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, var_name});

    Value *aux;
    for (int j=0; j<V_Dims.size(); j++)
    {
      aux = V_Dims[j]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
      Builder->CreateCall(TheModule->getFunction("StoreDimsOnDemand"),
                                                  {var_name, aux});
    }
    
    Builder->CreateCall(TheModule->getFunction("CreatePinnedTensorOnDemand"),
                                              {var_name, Builder->CreateGlobalString(TensorInit)});

 
  }


  return ConstantFP::get(*TheContext, APFloat(0.0));
}





Value *Conv2dExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));



  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();


    Value *var_name;
    var_name = Builder->CreateGlobalString(VarName);

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {first_arg, var_name});
                                            
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, var_name});
    
    


    std::cout << "Parsing Conv2d var for: " << VarName << "\n";

    Builder->CreateCall(TheModule->getFunction("CreateConv2dOnDemand"),
                                              {var_name, Builder->CreateGlobalString(TensorInit),
                                               C->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad), OC->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad), Ks->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad), Stride->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad),
                                               Padding->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad)});
  }
  return ConstantFP::get(*TheContext, APFloat(0.0));
}



Value *MaxPool2dExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));



  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    
    Value *var_name, *type;
    var_name = Builder->CreateGlobalString(VarName);
    type = Builder->CreateGlobalString(Type);

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {first_arg, var_name});
                                            
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, var_name});
    

    
    std::cout << "Parsing Conv2d var for: " << VarName << "\n";

    Builder->CreateCall(TheModule->getFunction("CreateMaxPool2dOnDemand"),
                                              {var_name, type,
                                               Ks->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad),
                                               Stride->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad),
                                               Padding->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad)});
  }
  return ConstantFP::get(*TheContext, APFloat(0.0));
}



Value *BatchNorm2dExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    
    Value *var_name, *type;
    var_name = Builder->CreateGlobalString(VarName);
    type = Builder->CreateGlobalString(Type);

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {first_arg, var_name});
                                            
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, var_name});
    

    
    std::cout << "Parsing Conv2d var for: " << VarName << "\n";

    Builder->CreateCall(TheModule->getFunction("CreateBatchNorm2dOnDemand"),
                                              {var_name, 
                                               C->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad)});
  }
  return ConstantFP::get(*TheContext, APFloat(0.0));
}




Value *BN2dReluExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    
    Value *var_name, *type;
    var_name = Builder->CreateGlobalString(VarName);
    type = Builder->CreateGlobalString(Type);

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {first_arg, var_name});
                                            
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, var_name});
    

    
    std::cout << "Parsing Conv2d var for: " << VarName << "\n";

    Builder->CreateCall(TheModule->getFunction("CreateBN2dReluOnDemand"),
                                              {var_name, C->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad)});
  }
  return ConstantFP::get(*TheContext, APFloat(0.0));
}


Value *LSTMExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));



  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();


    Value *var_name;
    var_name = Builder->CreateGlobalString(VarName);

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {first_arg, var_name});
                                            
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, var_name});
    
    


    std::cout << "Parsing Conv2d var for: " << VarName << "\n";

    Builder->CreateCall(TheModule->getFunction("CreateLSTMOnDemand"),
                                              {var_name, Builder->CreateGlobalString(TensorInit),
                                               C->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad), OC->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad)});
  }
  return ConstantFP::get(*TheContext, APFloat(0.0));
}



Value *EmbeddingExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));



  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();


    Value *var_name;
    var_name = Builder->CreateGlobalString(VarName);

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {first_arg, var_name});
                                            
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, var_name});
    
    


    std::cout << "Parsing Embedding var for: " << VarName << "\n";

    Builder->CreateCall(TheModule->getFunction("CreateEmbeddingOnDemand"),
                                              {var_name, Builder->CreateGlobalString(TensorInit),
                                               C->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad), OC->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad)});
  }
  return ConstantFP::get(*TheContext, APFloat(0.0));
}




Value *LinearExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));



  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();


    Value *var_name;
    var_name = Builder->CreateGlobalString(VarName);

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {first_arg, var_name});
                                            
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, var_name});
    
    
    int_vec *notators = SetNotators(Notators);

    

    std::cout << "Parsing MHSA var for: " << VarName << "\n";

    Builder->CreateCall(TheModule->getFunction("CreateLinearOnDemand"),
                                              {var_name, Builder->CreateGlobalString(TensorInit),
                                               C->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad),
                                               OC->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad),
                                               VoidPtr_toValue(notators)});
  }
  return ConstantFP::get(*TheContext, APFloat(0.0));
}




Value *MHSAExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));



  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();


    Value *var_name;
    var_name = Builder->CreateGlobalString(VarName);

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {first_arg, var_name});
                                            
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, var_name});
    
    int_vec *notators = SetNotators(Notators);


    std::cout << "Parsing MHSA var for: " << VarName << "\n";

    Builder->CreateCall(TheModule->getFunction("CreateMHSAOnDemand"),
                                              {var_name, Builder->CreateGlobalString(TensorInit),
                                               nh->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad),
                                               C->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad),
                                               T->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad),
                                               VoidPtr_toValue(notators)});
  }
  return ConstantFP::get(*TheContext, APFloat(0.0));
}


Value *ReluExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));


  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    
    Value *var_name, *type;
    var_name = Builder->CreateGlobalString(VarName);
    type = Builder->CreateGlobalString(Type);

    bool is_self = GetSelf();
    bool is_attr = GetIsAttribute();

    if (is_self||is_attr)
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {first_arg, var_name});
                                            
    if (!(is_self||is_attr))
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                            {scope_str, var_name});
    

    
    std::cout << "Parsing Conv2d var for: " << VarName << "\n";

    Builder->CreateCall(TheModule->getFunction("CreateReluOnDemand"),
                                              {var_name});
  }
  return ConstantFP::get(*TheContext, APFloat(0.0));
}


Value *CallExprAST::codegen(Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  if (not ShallCodegen)
    return ConstantFP::get(*TheContext, APFloat(0.0f));
  // Look up the name in the global module table.
  std::string tgt_function = Callee;
  

  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  std::string functionName = TheFunction->getName().str();
  std::string tgt_function_name;

  //std::cout << "\n\nFunction: " << tgt_function << "\n";

  int nested_function;
  if (functionName=="__anon_expr" || starts_with(functionName.c_str(), "__async_"))
  {
    nested_function=0;
  }
  else
    nested_function=1;


  bool has_scope = false;
  bool changed_first_arg = false;
  bool has_first_arg_copy = false;
  bool must_free_arg0 = false;

  Value *name;
  

  int thread = 0;

  //TODO: Solve scope_str discontinuity on async functions
  if (starts_with(functionName.c_str(), "__async_"))
  {
    std::cout << "\n\n\n\n\nASYNC" << "\n\n\n\n\n";
    scope_str = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});

    std::string copy = functionName;
    std::string prefix = "__async_";

    size_t pos = copy.find(prefix);
    copy.erase(pos, prefix.length());
    thread = std::stoi(copy);
    thread_id = ConstantInt::get(Type::getInt32Ty(*TheContext), thread);
    has_grad  = ConstantInt::get(Type::getInt32Ty(*TheContext), 1);
  }
  
  //std::cout << "\n\n\nFunction name: " << functionName << "\n";
  //std::cout << "THREAD IS: " << thread << "\n\n\n\n";



  //Builder->CreateCall(TheModule->getFunction("FreeChar"), {previous_scope});
  previous_scope = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                        {scope_str});


  Value *_pre_dot_str = Builder->CreateGlobalString(_pre_dot);
  Value *first_arg_copy;




  if (isAttribute && !isSelf && !in_str(tgt_function, native_methods))
  { // e.g: model.forward()
    if (nested_function)
    {
      first_arg_copy = Builder->CreateCall(TheModule->getFunction("CopyString"),
                                                    {first_arg});
      has_first_arg_copy = true;
    }
    
    
    first_arg = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                {previous_scope, _pre_dot_str});

    changed_first_arg = true;
  }
  
  
  

  int target_args_size = Args.size();
  std::vector<Value *> ArgsV;


  if (in_str(Callee, threaded_tensor_functions))
  {
    std::cout << "\n\n\n\n\nCALLEE " << Callee <<" IS IN A THREAD" << "\n\n\n\n\n";

    ArgsV.push_back(thread_id);

    target_args_size+=1;
  }
  



  bool is_self_of_nested_function = (nested_function==1 && isSelf);
  
  // Handle self or object attribute expressions
  if(isSelf || isAttribute)
  {
    bool not_coding_language_method = (!in_str(tgt_function, native_methods));    

    

    if (not_coding_language_method)
      tgt_function = Class+tgt_function;

    if (!is_self_of_nested_function && not_coding_language_method)
    {

      _pre_dot_str = Builder->CreateCall(TheModule->getFunction("ConcatScopeAtCallExpr"),
                {scope_str, _pre_dot_str});

      first_arg = Builder->CreateCall(TheModule->getFunction("FirstArgOnDemand"),
                                                    {first_arg,
                                                     _pre_dot_str,
                                                     Builder->CreateGlobalString(Class),
                                                     Builder->CreateGlobalString(Callee),
                                                     ConstantInt::get(Type::getInt32Ty(*TheContext), nested_function),
                                                     ConstantInt::get(Type::getInt32Ty(*TheContext), isSelf),
                                                     ConstantInt::get(Type::getInt32Ty(*TheContext), isAttribute)});
      
    }
    if (is_self_of_nested_function && not_coding_language_method)
    { // object method inside object method
      first_arg_copy = Builder->CreateCall(TheModule->getFunction("CopyString"), {first_arg});

      //first_arg = Builder->CreateCall(TheModule->getFunction("CopyString"), {first_arg});
      first_arg = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                                    {first_arg_copy,
                                                     _pre_dot_str});
      has_first_arg_copy = true;
    }
    changed_first_arg = not_coding_language_method;
    

    //name = NameSolver->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);
    
    if (CalleeOverride!="none"||in_str(Callee, native_methods))
    { // e.g: x.view()
    
      if (isSelf&&!isAttribute)
        ArgsV.push_back(first_arg);
      if (!isSelf&&isAttribute)
      {
        Value *arg = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                        {previous_scope, _pre_dot_str});
        ArgsV.push_back(arg);
        //must_free_arg0 = true; //TODO: break?
      }
      
      if (isSelf && isAttribute)
      { // e.g: self.can_load_.first_nonzero()
        // Extend first arg
        ArgsV.push_back(first_arg);
        ArgsV[0] = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                        {ArgsV[0], _pre_dot_str});
        //must_free_arg0 = true; //TODO: break?
        
      }

      if (in_str(Callee, return_tensor_methods))
      {
        ArgsV[1] = Builder->CreateCall(TheModule->getFunction("LoadTensor"), {ArgsV[1]});
        must_free_arg0 = false;
      }

    }
    else // Pass first_arg's reference for the derived AST nodes.
      ArgsV.push_back(first_arg);
    
    target_args_size+=1;
  }

  

  if (!(CalleeOverride!="none" || in_str(Callee, native_fn))||Callee=="print_scope") // user defined functions
  {
    has_scope = true;
    
    if(Callee!="print_scope")
      scope_str = Builder->CreateCall(TheModule->getFunction("RandomStrOnDemand"), {});
    
    
    ArgsV.push_back(scope_str); // Pass scope's reference for the derived AST nodes.
    ArgsV.push_back(previous_scope);
    ArgsV.push_back(thread_id);
    ArgsV.push_back(has_grad);
    
    target_args_size+=4;
  }

  if(in_str(tgt_function, require_scope_functions))
  {
    ArgsV.push_back(scope_str); // Pass scope's reference for the derived AST nodes.
    target_args_size+=1;
  }

  

  
  

  // Detect function errors
  Function *CalleeF;
  if (!IsVarForward)
  {
    CalleeF = getFunction(tgt_function);
    if (!CalleeF)
    {
      std::string _error = "The referenced function "+ tgt_function +" was not yet declared.";
      return LogErrorV(_error);
    }

    tgt_function_name = CalleeF->getName().str();

    // If argument mismatch error.
    if ((CalleeF->arg_size()) != target_args_size && !in_str(tgt_function_name, vararg_methods))
    {
      //std::cout << "CalleeF->arg_size() " << CalleeF->arg_size() << " target_args_size " << target_args_size << "\n";
      std::string _error = "Incorrect parameters used on function " + tgt_function + " call.";
      return LogErrorV(_error);
    }
  }
  //std::cout << "\n\n\nCalling function: " << tgt_function <<"\n";





  // Get Arguments
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    //std::cout << "\nCall codegen for argument n°: " << i << ".\n";

    // deal with firstarg on self.mcts(self.actions)
    Value *fa = (isAttribute && !isSelf && !in_str(tgt_function, native_methods) && nested_function) ? first_arg_copy : first_arg;
    //Value *fa = first_arg;

    //deal with scope on model.forward()
    Value *_scope = (!in_str(tgt_function, native_methods)) ? previous_scope : scope_str;
    

    Value * arg;
    //std::cout << "ARG: " << Args[i]->GetName() << " has self: " << Args[i]->GetSelf() << " and type: " << Args[i]->GetType() <<  "\n\n";
    if ((Args[i]->GetType()=="tensor" || Args[i]->GetType()=="pinned_tensor") && Args[i]->GetIsVarLoad())
    {
      //if (starts_with(functionName.c_str(), "__async_"))
      //  Builder->CreateStore(Builder->CreateGlobalString("threaded_"), _scope);
      VariableExprAST *Arg = static_cast<VariableExprAST *>(Args[i].get());
      arg = Arg->NameSolver->codegen(first_arg, _scope, previous_scope, thread_id, has_grad);

      arg = Builder->CreateCall(TheModule->getFunction("LoadTensor"), {arg});
    }
    else
      arg = Args[i]->codegen(fa, _scope, previous_scope, thread_id, has_grad);

  
    ArgsV.push_back(arg);


    if (!ArgsV.back())
      return nullptr;
  }



  
  Value *ret = ConstantFP::get(*TheContext, APFloat(0.0f));
  //std::cout << "\n\nCreate call: "  << tgt_function_name << " from parent: " << functionName << ", with override: " << CalleeOverride << "\n\n";

  if (CalleeOverride=="none")
    ret = Builder->CreateCall(CalleeF, ArgsV, "calltmp");
  else
  {
    if(in_str(CalleeOverride, threaded_tensor_functions))
      ArgsV.push_back(thread_id);
    
    //std::cout << "Override: " << CalleeOverride << "\n";
    if (in_str(CalleeOverride, native_modules))
    {
      CalleeF = getFunction(CalleeOverride);
      Value *conv_name = Builder->CreateGlobalString(tgt_function);
      Value *is_attr = ConstantInt::get(Type::getInt32Ty(*TheContext), (int)(isSelf));
      ArgsV.push_back(conv_name);
      ArgsV.push_back(is_attr);
      
      if (CalleeF->arg_size() != ArgsV.size())
      {
        std::string _error = "Incorrect parameters used on function " + tgt_function + " call.";
        return LogErrorV(_error);
      }
      ret = Builder->CreateCall(CalleeF, ArgsV, "calltmp");

    }
    else if (CalleeOverride=="SplitString")
    {
      Value *V = Builder->CreateCall(TheModule->getFunction("LoadStrOnDemand"), 
                                     {Builder->CreateGlobalString(PreDot)});
      
      ret = Builder->CreateCall(getFunction("SplitString"), 
                          {V, ArgsV[1]});

    }
    else if (CalleeOverride=="ToFloat")
    {
      //std::cout << "\n\nTO FLOAT HAS TYPE " << Args[0]->GetType() << "\n";
      if (Args[0]->GetType()=="str")
        ret = Builder->CreateCall(getFunction("StrToFloat"), 
                          {ArgsV[0]});

    } else
      ret = Builder->CreateCall(getFunction(CalleeOverride), ArgsV, "calltmp");
  }

  
  
  
  Builder->CreateCall(TheModule->getFunction("FreeChar"), {previous_scope});
  
  if (changed_first_arg)
    Builder->CreateCall(TheModule->getFunction("FreeChar"), {first_arg});
  
  if (has_first_arg_copy)
    Builder->CreateCall(TheModule->getFunction("FreeChar"), {first_arg_copy});

  if (has_scope)
    Builder->CreateCall(TheModule->getFunction("FreeChar"), {scope_str});

  if (must_free_arg0)
    Builder->CreateCall(TheModule->getFunction("FreeChar"), {ArgsV[0]});
  
  return ret;
}







Function *PrototypeAST::codegen() {
  if (not ShallCodegen)
    return nullptr;
  // Make the function type:  float(float,float) etc.

  std::vector<Type *> types;


  for (auto &type : Types)
  {
    if (type=="s"||type=="t"||type=="c")
      types.push_back(int8PtrTy);
    else if(type=="i")
      types.push_back(Type::getInt32Ty(*TheContext));
    else
      types.push_back(Type::getFloatTy(*TheContext));
  }


  FunctionType *FT = FunctionType::get(Type::getFloatTy(*TheContext), types, false);
  

  Function *F =
      Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

  // Set names for all arguments.
  unsigned Idx = 0;
  for (auto &Arg : F->args())
    Arg.setName(Args[Idx++]);
  

  return F;
}



extern "C" void *SplitString(char *self, char *pattern)
{

  //std::cout << "\n\nSPLITTING: " << self << ", with pattern: " << pattern << "\n";


  std::vector<char *> result;
  char *input = strdup(self); // Duplicate the input string to avoid modifying the original
  char *token = strtok(input, pattern); // Get the first token

  while (token != nullptr) {
    result.push_back(token);
    token = strtok(nullptr, pattern); // Get the next token
  }

  std::string random_str = RandomString(15);
  StrVecAuxHash[random_str] = result;
  AuxRandomStrs[random_str] = "str_vec";
    
  return &StrVecAuxHash[random_str];
    
}




// INDEX METHODS

extern "C" char *SplitStringIndexate(char *name, char *pattern, float idx)
{
  pthread_mutex_lock(&clean_scope_mutex);
  char *self = NamedStrs[name];
  pthread_mutex_unlock(&clean_scope_mutex);
  //std::cout << "splitting: " << self << ", with pattern: " << pattern << "\n";

  
  std::vector<char *> splits;
  char *input = (char*)malloc(strlen(self) + 1);
  memcpy(input, self, strlen(self) + 1);
  //strcpy(input, self);

  char *saveptr;
  char *token = strtok_r(input, pattern, &saveptr); // Get the first token

  while (token != nullptr) {
    splits.push_back(token);
    token = strtok_r(nullptr, pattern, &saveptr); // Get the next token
  }


  //std::cout << "splitting " << name << "\n";

  if(splits.size()<=1)
  {
    std::string _err = "\nFailed to split.";
    LogErrorS(_err);
    std::cout << "" << name << "\n";
    return nullptr;
  }

  if (idx < 0)
    idx = splits.size() + idx;
  
  //std::cout << "Spltting " << self << " with " << pattern <<" at ["<<idx<<"]:  " << splits[idx] << "\n";
 
  // Convert the retained token to a std::string
  char *result = splits[idx];

  delete[] name;
  delete[] input;

  return result;
}




extern "C" char *IndexStrVec(std::vector<char*> vec, float _idx)
{

  int idx = (int) _idx;

  //std::cout << "Str vec indexed at [" << idx << "]: " << vec[idx] << "\n";
  
  
  return vec[idx];
}


extern "C" char * IndexClassStrVec(char *vec_name, float _idx)
{
  int idx = (int) _idx;


  std::vector<char*> vec = ClassStrVecs[vec_name];

  //std::cout << "Class object Str Vec " << vec_name << "indexed at [" << idx << "]: " << vec[idx] << "\n";
  delete[] vec_name;

  return vec[idx];
}


extern "C" float IndexClassFloatVec(char *vec_name, float _idx)
{
  int idx = (int) _idx;

  float ret = ClassFloatVecs[vec_name][idx];
  delete[] vec_name;
  return ret;
}



extern "C" float StrToFloat(char *in_str)
{
  //std::cout << "\n\nstr to float of " << in_str << "\n\n\n";

  char *copied = (char*)malloc(strlen(in_str) + 1);
  strcpy(copied, in_str);
  char *end;

  float ret = std::strtof(copied, &end);
  delete[] copied;
  return ret;
}




extern "C" void objAttr_var_from_var(char *LName, char *RName)
{
  //std::cout << "objAttr_var_from_var of " << LName << " from " << RName << "\n";

  //std::cout << "Loading: " << NamedObjects[RName] << "\n";
  //std::cout << "Replacing: " << NamedObjects[LName] << "\n";

  NamedObjects[LName] = NamedObjects[RName];
  
  
}

extern "C" void objAttr_var_from_vec(char *LName, char *RName)
{
  //std::cout << "objAttr_var_from_vec of " << LName << " from " << RName << "\n";

  //std::cout << "Loading: " << objectVecs[RName] << "\n";
  //std::cout << "Replacing: " << NamedObjects[LName] << "\n";

  NamedObjects[LName] = objectVecs[RName];

  
}

extern "C" void objAttr_vec_from_var(char *LName, char *RName)
{
  //std::cout << "objAttr_vec_from_var of " << LName << " from " << RName << "\n";

  //std::cout << "Loading: " << NamedObjects[RName] << "\n";
  //std::cout << "Replacing: " << objectVecs[LName] << "\n";

  objectVecs[LName] = NamedObjects[RName];

  
}


extern "C" void objAttr_vec_from_vec(char *LName, char *RName)
{
  //std::cout << "objAttr_vec_from_vec of " << LName << " from " << RName << "\n";

  //std::cout << "Loading: " << objectVecs[RName] << "\n";
  //std::cout << "Replacing: " << objectVecs[LName] << "\n";

  objectVecs[LName] = objectVecs[RName];

  
}

extern "C" float append(char *self, char *obj_name)
{
  //char* copied = (char*)malloc(strlen(in_str) + 1);
  //strcpy(copied, in_str);

  std::cout << "\n\nAPPEND OF " << obj_name << " into: " << self << "\n";
  
  std::string obj_name_str = obj_name;


  


  int obj_vec_last_id = 0;
  if (objectVecsLastId.count(self)>0)
  {
    obj_vec_last_id = objectVecsLastId[self];
    obj_vec_last_id+=1;
  }
  objectVecsLastId[self] = obj_vec_last_id;

  std::string indexed_self = self + std::to_string(obj_vec_last_id);
  objectVecs[indexed_self] = NamedObjects[obj_name];

  
  

  return 0;
}

extern "C" char *LoadObjectScopeName(char *self)
{
  if (objectVecs.count(self)==0)
  {
    std::string _self = self;
    std::string _error = "Object "+_self+" does not exist";
    LogErrorS(_error);
    return "";
  }

  /*
  for (auto &pair : objectVecs)
  {
    std::cout <<  pair.first << ": " << pair.second << "\n";
  }
  */
  std::string ret = objectVecs[self];
  if(ret.length()==0)
  {
    for (auto &pair : objectVecs)
      std::cout <<  pair.first << ": " << pair.second << "\n";

    std::string _self = self;
    std::string _error = "Loaded object "+_self+" has zero length.";
    LogErrorS(_error);
  }


  //std::cout << "LoadObjectScopeName is: " << ret << ", from self: " << self << "\n";

  delete[] self;

  return str_to_char(ret);
}




const PrototypeAST& FunctionAST::getProto() const {
  return *Proto;
}

const std::string& FunctionAST::getName() const {
  return Proto->getName();
}

Function *FunctionAST::codegen() {
  if (not ShallCodegen)
    return nullptr;
  
  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below.
  auto &P = *Proto;

    

  FunctionProtos[Proto->getName()] = std::move(Proto);
  std::string function_name = P.getName();

  Function *TheFunction = getFunction(function_name);
  if (!TheFunction)
    return nullptr;

  // If this is an operator, install it.
  if (P.isBinaryOp())
    BinopPrecedence[P.getOperatorName()] = P.getBinaryPrecedence();

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(*TheContext, "entry", TheFunction);
  Builder->SetInsertPoint(BB);

  
  // Record the function arguments in the NamedValues map.

  Value *first_arg, *scope_str, *previous_scope, *thread_id, *has_grad;
  /*
  first_arg = Builder->CreateAlloca(int8PtrTy);
  scope_str = Builder->CreateAlloca(int8PtrTy);
  previous_scope = Builder->CreateAlloca(int8PtrTy);
  */

  


  thread_id = ConstantInt::get(Type::getInt32Ty(*TheContext), 0);
  has_grad  = ConstantInt::get(Type::getInt32Ty(*TheContext), 1);
  if (function_name=="__anon_expr")
  {
    first_arg = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});
    scope_str = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});
    previous_scope = Builder->CreateCall(TheModule->getFunction("GetEmptyChar"), {});
  }
  


  std::cout << "\033[32mExecuting function: " << function_name << " \033[0m\n";

  NamedValues.clear();

  bool has_self = false; 
  bool has_scope = false;
  bool has_previous_scope = false;
  

  float val;
  int i = 0;
  for (auto &Arg : TheFunction->args()) {
    // Create an alloca for this variable.
    
    
    std::string arg_name = Arg.getName().str();
    //std::cout << "FUNCTION ARG IS: " << arg_name  << "\n";

    std::string __print = "FUNCTION ALLOCA OF " + std::string(Arg.getName()) + " ";


    // Default args
    if (arg_name == "self")
    {
      first_arg = Builder->CreateCall(TheModule->getFunction("CopyString"), {&Arg});
      
      has_self = true;
    }
    if (arg_name == "scope_str")
    {
      scope_str = Builder->CreateCall(TheModule->getFunction("CopyString"), {&Arg});
      
      has_scope = true;
    }
    if (arg_name == "previous_scope")
    {
      previous_scope = Builder->CreateCall(TheModule->getFunction("CopyString"), {&Arg});
      
      has_previous_scope = true;
    }
    if (arg_name == "thread_id")
      thread_id = &Arg;
    if (arg_name == "has_grad")
      has_grad = &Arg;
    

    // Coder args
    if (in_str(arg_name, floatVars))
    {
      Value *var_name = Builder->CreateGlobalString(arg_name);
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                    {scope_str, var_name});

      Builder->CreateCall(TheModule->getFunction("StoreArgOnDemand"),
                                                  {scope_str, var_name, &Arg});
    } else if (in_str(arg_name, strVars))
    {
      Value *var_name = Builder->CreateGlobalString(arg_name);
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"), //TODO: Store scope vars to clean for this too
                                    {scope_str, var_name});


      Builder->CreateCall(TheModule->getFunction("StoreStrOnDemand"),
                                                  {var_name, &Arg});
    }
    else if (!in_str(arg_name, tensorVars))
    {
      AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, Arg.getName());
      Builder->CreateStore(&Arg, Alloca);
      //Builder->CreateStore(Builder->CreateLoad(Type::getFloatTy(*TheContext), &Arg), Alloca);



      NamedValues[std::string(Arg.getName())] = Alloca;

    }
    else
    {
      if (in_str(arg_name, tensorVars))
      {
        //Builder->CreateCall(TheModule->getFunction("print"),
        //  {Builder->CreateGlobalString(__print), &Arg});

        Builder->CreateCall(TheModule->getFunction("CopyArgTensor"),
                          {&Arg,
                           Builder->CreateGlobalString(arg_name),
                           previous_scope,
                           scope_str,
                           thread_id});
      }
    }
  }
  


  Value *RetVal;
  for (auto &body : Body)
    RetVal = body->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad);



  //Value *aux = Builder->CreateGlobalString(function_name);


  
  


  
  if(has_self)
    Builder->CreateCall(TheModule->getFunction("FreeChar"), {first_arg});

  if(has_scope)
  {
    Builder->CreateCall(TheModule->getFunction("CleanScopeVars"), {scope_str, thread_id});
    Builder->CreateCall(TheModule->getFunction("FreeChar"), {scope_str}); 
  }

  
  if(has_previous_scope)
    Builder->CreateCall(TheModule->getFunction("FreeChar"), {previous_scope});

  
  

  if (RetVal) {
    // Finish off the function.
    
    
    Builder->CreateRet(RetVal);
    

    // Validate the generated code, checking for consistency.
    verifyFunction(*TheFunction);


    return TheFunction;
  }


  // Error reading body, remove function.
  TheFunction->eraseFromParent();

  if (P.isBinaryOp())
    BinopPrecedence.erase(P.getOperatorName());
  return nullptr;
}





//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//


static void InitializeModule() {
  //std::cout << "\nINITIALIZING A NEW MODULE"  << "\n\n";

  // Open a new context and module.
  TheContext = std::make_unique<LLVMContext>();
  TheModule = std::make_unique<Module>("my cool jit", *TheContext);
  TheModule->setDataLayout(TheJIT->getDataLayout());

  //std::cout << "Initialize Module\n";
  

  // Create a new builder for the module.
  Builder = std::make_unique<IRBuilder<>>(*TheContext);

  floatPtrTy = Type::getFloatTy(*TheContext)->getPointerTo();
  int8PtrTy = Type::getInt8Ty(*TheContext)->getPointerTo();
  ShallCodegen = true;
  seen_var_attr = false;



  //===----------------------------------------------------------------------===//
  // Scalar   Operations
  //===----------------------------------------------------------------------===//

  // 
  FunctionType *fmaxTy = FunctionType::get( //TODO: automatic type detection for max and min
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("max", fmaxTy);


  // 
  FunctionType *fminTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("min", fminTy);


  // 
  FunctionType *flog2Ty = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("logE2f", flog2Ty);


  // 
  FunctionType *roundTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("roundE", roundTy);


  // 
  FunctionType *logical_notTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("logical_not", logical_notTy);


  // 
  FunctionType *dir_existsTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("dir_exists", dir_existsTy);


  // 
  FunctionType *path_existsTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("path_exists", path_existsTy);


  // 
  FunctionType *floorTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("floorE", floorTy);


  //===----------------------------------------------------------------------===//
  // Tensor -- Scalar   Operations
  //===----------------------------------------------------------------------===//

  //
  FunctionType *CudaScalarMultTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("CudaScalarMult", CudaScalarMultTy);


  //
  FunctionType *CudaScalarDivTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarDiv", CudaScalarDivTy);


  //
  FunctionType *CudaReverseScalarDivTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaReverseScalarDiv", CudaReverseScalarDivTy);


  //
  FunctionType *CudaScalarAddTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarAdd", CudaScalarAddTy);


  //
  FunctionType *CudaScalarSubTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarSub", CudaScalarSubTy);


  //
  FunctionType *CudaScalarEqualTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarEqual", CudaScalarEqualTy);


  //
  FunctionType *CudaScalarDiffTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarDiff", CudaScalarDiffTy);


  //
  FunctionType *CudaScalarMinorTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarMinor", CudaScalarMinorTy);


  //
  FunctionType *CudaScalarHigherTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarHigher", CudaScalarHigherTy);

  
  //
  FunctionType *CudaScalarHigherEqTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarHigherEq", CudaScalarHigherEqTy);


  //
  FunctionType *CudaScalarMinorEqTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaScalarMinorEq", CudaScalarMinorEqTy);

  

  //===----------------------------------------------------------------------===//
  // Tensor Tensor CUDA Ops
  //===----------------------------------------------------------------------===//

  //
  FunctionType *CudaMultTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext),
       int8PtrTy,
       int8PtrTy,
       Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaMult", CudaMultTy);


  //
  FunctionType *CudaAddTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext),
       int8PtrTy,
       int8PtrTy, Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaAdd", CudaAddTy);


  //
  FunctionType *CudaSubTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext),
       int8PtrTy,
       int8PtrTy, Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaSub", CudaSubTy);


  //
  FunctionType *CudaEqualTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext),
       int8PtrTy,
       int8PtrTy, Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaEqual", CudaEqualTy);


  //
  FunctionType *CudaHadamardTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext),
       int8PtrTy,
       int8PtrTy, Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CudaHadamard", CudaHadamardTy);


  //
  FunctionType *CudaDivTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext),
       int8PtrTy,
       int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("CudaDiv", CudaDivTy);


  //
  FunctionType *LoadTensorTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("LoadTensor", LoadTensorTy);


  //
  FunctionType *IdxTensorTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)}, 
      true // vararg
  );
  TheModule->getOrInsertFunction("IdxTensor", IdxTensorTy);


  //
  FunctionType *AttrPinnedFromTensorOnIdxTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)}, 
      true // vararg
  );
  TheModule->getOrInsertFunction("AttrPinnedFromTensorOnIdx", AttrPinnedFromTensorOnIdxTy);


  //
  FunctionType *IdxTensorWithTensorTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("IdxTensorWithTensor", IdxTensorWithTensorTy);


  //
  FunctionType *PrintTensorFTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {floatPtrTy,
       Type::getInt32Ty(*TheContext),
       Type::getInt32Ty(*TheContext),}, 
      false
  );
  TheModule->getOrInsertFunction("PrintTensorF", PrintTensorFTy);


  //
  FunctionType *print_tensorTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("print_tensor", print_tensorTy);


  //
  FunctionType *LoadDimsTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("LoadDims", LoadDimsTy);


  //
  FunctionType *PrintDimsTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("PrintDims", PrintDimsTy);
  

  //
  FunctionType *clipTy = FunctionType::get(
      floatPtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy,
       Type::getInt32Ty(*TheContext),
       Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("clip", clipTy);


  //
  FunctionType *network_emaTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("network_ema", network_emaTy);


  //===----------------------------------------------------------------------===//
  // Backward and Optimizers CUDA Ops
  //===----------------------------------------------------------------------===//

  //
  FunctionType *BackpropagationTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {}, 
      false
  );
  TheModule->getOrInsertFunction("backprop", BackpropagationTy);


  //
  FunctionType *clean_forwardTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("clean_forward", clean_forwardTy);
    
  
  //
  FunctionType *SGDTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("SGD", SGDTy);
  
  
  //
  FunctionType *AdamWTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("AdamW", AdamWTy);
  
  
  //
  FunctionType *OneCycleLRTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("OneCycleLR", OneCycleLRTy);
  
  
  //
  FunctionType *CosineLRTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("CosineLR", CosineLRTy);



  //===----------------------------------------------------------------------===//
  // Unary CUDA Ops
  //===----------------------------------------------------------------------===//

  //
  FunctionType *CudaLogTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("logE", CudaLogTy);
  

  // 
  FunctionType *log2Ty = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("logE2", log2Ty);


  // 
  FunctionType *btc_multTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("btc_mult", btc_multTy);


  // 
  FunctionType *btc_multTTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("btc_multT", btc_multTTy);


  // 
  FunctionType *softmaxTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("softmax", softmaxTy);


  // 
  FunctionType *priority_sampleTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("priority_sample", priority_sampleTy);


  // 
  FunctionType *priority_sample_valTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("priority_sample_val", priority_sample_valTy);


  // 
  FunctionType *importance_sample_idxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("importance_sample_idx", importance_sample_idxTy);


  // 
  FunctionType *importance_sample_weightTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("importance_sample_weight", importance_sample_weightTy);


  // 
  FunctionType *self_attnTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("self_attn", self_attnTy);
  

  //
  FunctionType *reluTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("relu", reluTy);
  

  //
  FunctionType *gatherTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("gather", gatherTy);
  

  //
  FunctionType *rl_discounted_returnTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("rl_discounted_return", rl_discounted_returnTy);
  

  //
  FunctionType *geluTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("gelu", geluTy);
  

  //
  FunctionType *sigmoidTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("sigmoid", sigmoidTy);
  

  //
  FunctionType *sigmoid_add2weightsTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("sigmoid_add2weights", sigmoid_add2weightsTy);
  

  //
  FunctionType *tanhTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("_tanh", tanhTy);


  //
  FunctionType *conv2dForwardTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("ConvForward2d", conv2dForwardTy);


  //
  FunctionType *LSTMForwardTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("LSTMForward", LSTMForwardTy);


  //
  FunctionType *MHSAForwardTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("MHSAForward", MHSAForwardTy);


  //
  FunctionType *LinearForwardTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("LinearForward", LinearForwardTy);


  //
  FunctionType *EmbeddingForwardTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("EmbeddingForward", EmbeddingForwardTy);


  //
  FunctionType *MaxPoolForward2dTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("MaxPoolForward2d", MaxPoolForward2dTy);


  //
  FunctionType *BatchNormForward2dTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("BatchNormForward2d", BatchNormForward2dTy);


  //
  FunctionType *BN2dReluForwardTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("BN2dReluForward", BN2dReluForwardTy);


  //
  FunctionType *ReluForwardTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("ReluForward", ReluForwardTy);


  //
  FunctionType *cropTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("RandomCrop", cropTy);


  //
  FunctionType *RandomHorizontalFlipTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("RandomHorizontalFlip", RandomHorizontalFlipTy);


  //
  FunctionType *NormalizeImgTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("NormalizeImg", NormalizeImgTy);


  //
  FunctionType *JitterTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("Jitter", JitterTy);


  //
  FunctionType *dropoutTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("dropout", dropoutTy);
  

  //
  FunctionType *onehotTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("onehot", onehotTy);
  

  //
  FunctionType *shapeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("shape", shapeTy);
  

  //
  FunctionType *printttTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("printtt", printttTy);
  

  //
  FunctionType *evalTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {},
      false
  );
  TheModule->getOrInsertFunction("eval", evalTy);
  

  //
  FunctionType *trainTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {},
      false
  );
  TheModule->getOrInsertFunction("train", trainTy);

  
  //
  FunctionType *repeat_interleaveTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("repeat_interleave", repeat_interleaveTy);
  

  // 
  FunctionType *sumTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("sum", sumTy);
  

  // 
  FunctionType *prodTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("prod", prodTy);


  // 
  FunctionType *meanTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("mean", meanTy);
  

  // 
  FunctionType *maxTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("tmax", maxTy);


  //
  FunctionType *argmaxTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("argmax", argmaxTy);
  

  //
  FunctionType *topkTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("topk", topkTy);


  //
  FunctionType *viewTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext),int8PtrTy,int8PtrTy,Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("view", viewTy);


  //
  FunctionType *CalculateIdxOffsetTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("CalculateIdxOffset", CalculateIdxOffsetTy);


  //
  FunctionType *NewVecToTensorTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext),Type::getFloatTy(*TheContext)},
      true // vararg
  );
  TheModule->getOrInsertFunction("NewVecToTensor", NewVecToTensorTy);
  

  //===----------------------------------------------------------------------===//
  // Loss CUDA Ops
  //===----------------------------------------------------------------------===//

  //
  FunctionType *cross_entropyTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("cross_entropy", cross_entropyTy);


  //
  FunctionType *cross_entropy_idxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("cross_entropy_idx", cross_entropy_idxTy);


  //
  FunctionType *mseTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("mse", mseTy);


  //
  FunctionType *mse_with_prioritiesTy = FunctionType::get(
      int8PtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy}, 
      false
  );
  TheModule->getOrInsertFunction("mse_with_priorities", mse_with_prioritiesTy);
  

  //===----------------------------------------------------------------------===//
  // File Handling Ops
  //===----------------------------------------------------------------------===//
  
  //
  FunctionType *load_imgTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("load_img", load_imgTy);
  

  //
  FunctionType *load_preprocess_imgTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("load_preprocess_img", load_preprocess_imgTy);
  

  //===----------------------------------------------------------------------===//
  // Pinned Tensor Ops
  //===----------------------------------------------------------------------===//

  //
  FunctionType *gload_imgTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("gload_img", gload_imgTy);


  //
  FunctionType *wload_imgTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("wload_img", wload_imgTy);


  //
  FunctionType *load_binTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("load_bin", load_binTy);


  //
  FunctionType *wload_binTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("wload_bin", wload_binTy);


  //
  FunctionType *load_bin_idxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      true
  );
  TheModule->getOrInsertFunction("load_bin_idx", load_bin_idxTy);


  //
  FunctionType *save_as_binTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("save_as_bin", save_as_binTy);


  //
  FunctionType *save_as_intTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("save_as_int", save_as_intTy);


  //
  FunctionType *wload_img_resizeTy = FunctionType::get(
      floatPtrTy,
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("wload_img_resize", wload_img_resizeTy);


  //
  FunctionType *save_imgTy = FunctionType::get(
      floatPtrTy,
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("save_img", save_imgTy);
  

  //
  FunctionType *AttrPinnedOnIdxTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("AttrPinnedOnIdx", AttrPinnedOnIdxTy);


  //
  FunctionType *gpuTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("gpu", gpuTy);


  //  
  FunctionType *gpuwTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("gpuw", gpuwTy);


  //===----------------------------------------------------------------------===//
  // Parallel Ops
  //===----------------------------------------------------------------------===//

  //  
  FunctionType *sleepTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("sleep", sleepTy);

  
  FunctionType *silent_sleepTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("silent_sleep", silent_sleepTy);


  //  
  FunctionType *start_timerTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("start_timer", start_timerTy);


  //  
  FunctionType *end_timerTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("end_timer", end_timerTy);
  


  auto pthreadPtr = Type::getInt8Ty(*GlobalContext)->getPointerTo();
  auto pthreadPtrTy = pthreadPtr->getPointerTo();

  // (void *) fn (void * arg)
  FunctionType *funVoidPtrint8PtrTy = FunctionType::get(
    int8PtrTy, {int8PtrTy},
    false);
  // int pthread_create(pthread_t * thread, const pthread_attr_t * attr,
  //                  void * (*start_routine)(void *), void * arg)
  // using void * in place of pthread_attr_t *
  FunctionType *pthreadCreateTy = FunctionType::get(
                                      Type::getVoidTy(*TheContext),
                                      {pthreadPtrTy,
                                       int8PtrTy,
                                       (funVoidPtrint8PtrTy)->getPointerTo(),
                                       int8PtrTy},
                                      false
                                    );
  TheModule->getOrInsertFunction("pthread_create_aux", pthreadCreateTy);


  // int pthread_join(pthread_t thread, void **value_ptr)
  FunctionType *pthreadJoinTy = FunctionType::get(
    Type::getVoidTy(*TheContext),
    {pthreadPtr},
    false);
  TheModule->getOrInsertFunction("pthread_join_aux", pthreadJoinTy);

  

  FunctionType *LockTy = FunctionType::get(
    Type::getVoidTy(*TheContext),
    {int8PtrTy},
    false);
  TheModule->getOrInsertFunction("LockMutex", LockTy);

  FunctionType *UnlockMutexTy = FunctionType::get(
    Type::getVoidTy(*TheContext),
    {int8PtrTy},
    false);
  TheModule->getOrInsertFunction("UnlockMutex", UnlockMutexTy);


  //===----------------------------------------------------------------------===//
  // Str Ops
  //===----------------------------------------------------------------------===//


  // char *
  FunctionType *globTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("_glob_b_", globTy);


  //
  FunctionType *zeros_vecTy = FunctionType::get(
      int8PtrTy,
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("zeros_vec", zeros_vecTy);


  //
  FunctionType *to_stringTy = FunctionType::get(
      int8PtrTy,
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("to_string", to_stringTy);


  //
  FunctionType *cat_str_floatTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("cat_str_float", cat_str_floatTy);


  //
  FunctionType *ones_vecTy = FunctionType::get(
      int8PtrTy,
      {Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("ones_vec", ones_vecTy);
  

  FunctionType *PrintFloatTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("PrintFloat", PrintFloatTy);

  FunctionType *UnbugFloatTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("UnbugFloat", UnbugFloatTy);


FunctionType *unbugTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {},
      false 
  );
  TheModule->getOrInsertFunction("unbug", unbugTy);
  
  
  FunctionType *SplitStringTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("SplitString", SplitStringTy);


  //
  FunctionType *SplitStringIndexateTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("SplitStringIndexate", SplitStringIndexateTy);


  //
  FunctionType *StrToFloatTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("StrToFloat", StrToFloatTy);


  //
  FunctionType *CopyStringTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("CopyString", CopyStringTy);


  //
  FunctionType *appendTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("append", appendTy);


  //
  FunctionType *objAttr_var_from_varTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("objAttr_var_from_var", objAttr_var_from_varTy);


  //
  FunctionType *objAttr_var_from_vecTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("objAttr_var_from_vec", objAttr_var_from_vecTy);


  //
  FunctionType *objAttr_vec_from_varTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("objAttr_vec_from_var", objAttr_vec_from_varTy);


  //
  FunctionType *objAttr_vec_from_vecTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("objAttr_vec_from_vec", objAttr_vec_from_vecTy);


  //
  FunctionType *LoadObjectScopeNameTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("LoadObjectScopeName", LoadObjectScopeNameTy);



  //
  FunctionType *PrintStrTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("PrintStr", PrintStrTy);


  //
  FunctionType *PrintStrVecTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("PrintStrVec", PrintStrVecTy);


  //
  FunctionType *PrintFloatVecTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("PrintFloatVec", PrintFloatVecTy);


  //
  FunctionType *print_scopeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)}, 
      false
  );
  TheModule->getOrInsertFunction("print_scope", print_scopeTy);


  //
  FunctionType *first_nonzeroTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("first_nonzero", first_nonzeroTy);


  //
  FunctionType *LoadStrVecOnDemandTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("LoadStrVecOnDemand", LoadStrVecOnDemandTy);


  //
  FunctionType *LoadFloatVecOnDemandTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("LoadFloatVecOnDemand", LoadFloatVecOnDemandTy);

  
  //
  FunctionType *StoreStrOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("StoreStrOnDemand", StoreStrOnDemandTy);

  
  //
  FunctionType *LoadStrOnDemandTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("LoadStrOnDemand", LoadStrOnDemandTy);


  //
  FunctionType *StoreStrVecOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("StoreStrVecOnDemand", StoreStrVecOnDemandTy);
  
  
  //
  FunctionType *StoreFloatVecOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("StoreFloatVecOnDemand", StoreFloatVecOnDemandTy);

  FunctionType *StoreFloatVecOnDemandOnIdxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("StoreFloatVecOnDemandOnIdx", StoreFloatVecOnDemandOnIdxTy);


  //
  FunctionType *LenStrVecTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("LenStrVec", LenStrVecTy);


  //
  FunctionType *ShuffleStrVecTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ShuffleStrVec", ShuffleStrVecTy);


  //
  FunctionType *IndexStrVecTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("IndexStrVec", IndexStrVecTy);


  //
  FunctionType *IndexClassStrVecTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("IndexClassStrVec", IndexClassStrVecTy);

  //
  FunctionType *IndexClassFloatVecTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("IndexClassFloatVec", IndexClassFloatVecTy);


  //
  FunctionType *AuxFnTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy}, 
      false 
  );
  TheModule->getOrInsertFunction("AuxFn", AuxFnTy);


  // char *
  FunctionType *shuffle_strTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("shuffle_str", shuffle_strTy);

  
  //
  FunctionType *TokenizeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("build_vocab", TokenizeTy);

  
  //
  FunctionType *tokenizeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("tokenize", tokenizeTy);

  
  //
  FunctionType *wtokenizeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("wtokenize", wtokenizeTy);

  
  //
  FunctionType *wtokenize_pad_leftTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("wtokenize_pad_left", wtokenize_pad_leftTy);

  
  //
  FunctionType *wtokenize_pad_left_batch_firstTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("wtokenize_pad_left_batch_first", wtokenize_pad_left_batch_firstTy);

  
  //
  FunctionType *wtokenize_pad_left_idxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("wtokenize_pad_left_idx", wtokenize_pad_left_idxTy);

  
  //
  FunctionType *write_zeroswTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("write_zerosw", write_zeroswTy);


  //
  FunctionType *InitObjectVecWithNullTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("InitObjectVecWithNull", InitObjectVecWithNullTy);


  //
  FunctionType *is_nullTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("is_null", is_nullTy);


  //===----------------------------------------------------------------------===//
  // Other Ops
  //===----------------------------------------------------------------------===//


  // 
  FunctionType *FirstArgOnDemandTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("FirstArgOnDemand", FirstArgOnDemandTy);
  

  // 
  FunctionType *objHashTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("objHash", objHashTy);
  

  // 
  FunctionType *LoadObjectTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("LoadObject", LoadObjectTy);
  

  //
  FunctionType *InstantiateObjectTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("InstantiateObject", InstantiateObjectTy);
  

  //
  FunctionType * GetEmptyCharTy = FunctionType::get(
      int8PtrTy,
      {},
      false 
  );
  TheModule->getOrInsertFunction("GetEmptyChar", GetEmptyCharTy);
  

  //
  FunctionType *FreeCharTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("FreeChar", FreeCharTy);
  

  //
  FunctionType *FreeCharFromFuncTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("FreeCharFromFunc", FreeCharFromFuncTy);
  

  //
  FunctionType * ConcatStrTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ConcatStr", ConcatStrTy);
  

  //
  FunctionType * ConcatStrFreeLeftTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ConcatStrFreeLeft", ConcatStrFreeLeftTy);
  

  //
  FunctionType * ConcatStrFreeRightTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ConcatStrFreeRight", ConcatStrFreeRightTy);
  

  //
  FunctionType * ConcatStrFreeTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ConcatStrFree", ConcatStrFreeTy);

  
  //
  FunctionType * ConcatNumToStrTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("ConcatNumToStr", ConcatNumToStrTy);

  
  //
  FunctionType * ConcatNumToStrFreeTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("ConcatNumToStrFree", ConcatNumToStrFreeTy);
  

  //
  FunctionType * ConcatScopeStrTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ConcatScopeStr", ConcatScopeStrTy);
  

  //
  FunctionType * ConcatScopeAtCallExprTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("ConcatScopeAtCallExpr", ConcatScopeAtCallExprTy);

  
  //
  FunctionType * AddToScopeCleanListTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("AddToScopeCleanList", AddToScopeCleanListTy);

  
  //
  FunctionType * AddFloatToScopeCleanListTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("AddFloatToScopeCleanList", AddFloatToScopeCleanListTy);

  
  //
  FunctionType *CleanScopeVarsTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("CleanScopeVars", CleanScopeVarsTy);
  

  //
  FunctionType * RandomStrOnDemandTy = FunctionType::get(
      int8PtrTy,
      {},
      false
  );
  TheModule->getOrInsertFunction("RandomStrOnDemand", RandomStrOnDemandTy);

  
  // 
  FunctionType *StoreOnDemandTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false //
  );
  TheModule->getOrInsertFunction("StoreOnDemand", StoreOnDemandTy);

  
  // 
  FunctionType *StoreOnDemandNoFreeTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false //
  );
  TheModule->getOrInsertFunction("StoreOnDemandNoFree", StoreOnDemandNoFreeTy);

  
  // 
  FunctionType *StoreArgOnDemandTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false //
  );
  TheModule->getOrInsertFunction("StoreArgOnDemand", StoreArgOnDemandTy);
  

  //
  FunctionType *LoadOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("LoadOnDemand", LoadOnDemandTy);
  

  //
  FunctionType *LoadOnDemandNoFreeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("LoadOnDemandNoFree", LoadOnDemandNoFreeTy);
  

  //
  FunctionType *StoreDimsOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("StoreDimsOnDemand", StoreDimsOnDemandTy);
  

  //
  FunctionType *CreatePinnedTensorOnDemandTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("CreatePinnedTensorOnDemand", CreatePinnedTensorOnDemandTy);


  //
  FunctionType *CreateTensorOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext), int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("CreateTensorOnDemand", CreateTensorOnDemandTy);


  //
  FunctionType *print_randomsTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("print_randoms", print_randomsTy);
  

  // 
  FunctionType *randintTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("randint", randintTy);


  //
  FunctionType *CreateConv2dOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy,
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("CreateConv2dOnDemand", CreateConv2dOnDemandTy);


  //
  FunctionType *CreateLSTMOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy,
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("CreateLSTMOnDemand", CreateLSTMOnDemandTy);


  //
  FunctionType *CreateEmbeddingOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy,
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("CreateEmbeddingOnDemand", CreateEmbeddingOnDemandTy);


  //
  FunctionType *CreateLinearOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy,
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       int8PtrTy,},
      false
  );
  TheModule->getOrInsertFunction("CreateLinearOnDemand", CreateLinearOnDemandTy);


  //
  FunctionType *CreateMHSAOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy,
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("CreateMHSAOnDemand", CreateMHSAOnDemandTy);


  //
  FunctionType *CreateBatchNorm2dOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("CreateBatchNorm2dOnDemand", CreateBatchNorm2dOnDemandTy);


  //
  FunctionType *CreateBN2dReluOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("CreateBN2dReluOnDemand", CreateBN2dReluOnDemandTy);


  //
  FunctionType *CreateReluOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy},
      false
  );
  TheModule->getOrInsertFunction("CreateReluOnDemand", CreateReluOnDemandTy);


  //
  FunctionType *CreateMaxPool2dOnDemandTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy,
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext),
       Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("CreateMaxPool2dOnDemand", CreateMaxPool2dOnDemandTy);


  //
  FunctionType *CopyArgTensorTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("CopyArgTensor", CopyArgTensorTy);

  
  //
  FunctionType *RemoveTensorScopeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("RemoveTensorScope", RemoveTensorScopeTy);


  //
  FunctionType *RemoveTensorScopeAttrOnIndexTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy,
       int8PtrTy,
       int8PtrTy,
       Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("RemoveTensorScopeAttrOnIndex", RemoveTensorScopeAttrOnIndexTy);


  //
  FunctionType *AttrTensorTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("AttrTensor", AttrTensorTy);

  FunctionType *AttrTensorNoFreeTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, floatPtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("AttrTensorNoFree", AttrTensorNoFreeTy);
  

  //
  FunctionType *AttrTensorOnIdxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       int8PtrTy,
       Type::getFloatTy(*TheContext),
       Type::getInt32Ty(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("AttrTensorOnIdx", AttrTensorOnIdxTy);
  

  //
  FunctionType *AttrTensorOnIdxTensorTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("AttrTensorOnIdxTensor", AttrTensorOnIdxTensorTy);
  

  //
  FunctionType *cpuTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("cpu", cpuTy);
  

  //
  FunctionType *cpu_idxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy,
       Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("cpu_idx", cpu_idxTy);


  //
  FunctionType *printTTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {Type::getInt32Ty(*TheContext), int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("PrintTensor", printTTy);
  
  
  //
  FunctionType *randu_likeTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("randu_like", randu_likeTy);

  
  //
  FunctionType *printTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext)}, 
      false 
  );
  TheModule->getOrInsertFunction("print", printTy);
  

}




ThreadSafeModule irgenAndTakeOwnership(FunctionAST &FnAST,
                                       const std::string &Suffix) {
  if (auto *F = FnAST.codegen()) {
    F->setName(F->getName() + Suffix);
    auto TSM = ThreadSafeModule(std::move(TheModule), std::move(TheContext));
    // Start a new module.
    InitializeModule();
    return TSM;
  } else
    report_fatal_error("Não foi possível compilar a função JIT de forma lazy");
}




static void HandleClass() { ParseClass(); }

static void HandleDefinition() {
  
  if (auto FnAST = ParseDefinition()) {

    FunctionProtos[FnAST->getProto().getName()] =
      std::make_unique<PrototypeAST>(FnAST->getProto());

    ExitOnErr(TheJIT->addAST(std::move(FnAST)));
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern() {
  if (auto ProtoAST = ParseExtern()) {
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
    
    fprintf(stderr, "%.2f\n", fp);

    // Delete the anonymous expression module from the JIT.
    ExitOnErr(RT->remove());    
}



static void HandleTopLevelExpression() {
  // Evaluate a top-level expression into an anonymous function.
  
  if (std::unique_ptr<FunctionAST> FnAST = ParseTopLevelExpr()) {
    CodegenTopLevelExpression(std::ref(FnAST));

  
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

/// top ::= definition | external | expression | ';'
static void MainLoop() {
  while (true) {
    //if (CurTok!=tok_space)
    //  std::cout << "MAIN LOOP, reading token: " << ReverseToken(CurTok) << "\n";
    

    switch (CurTok) {
      case tok_eof:
        return;
      case ';': // ignore top-level semicolons.
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
      case tok_extern:
        HandleExtern();
        break;
      default:
        HandleTopLevelExpression();
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

int main() {

  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
  cudaGetDeviceProperties(&deviceProp, deviceIdx);

  std::cout << "CuDNN Version: " << CUDNN_MAJOR << "." << CUDNN_MINOR << "." << CUDNN_PATCHLEVEL << std::endl;
  printf("Device %d: %s\n", deviceIdx, deviceProp.name);
  std::cout << "Device Max Compute Capability (SM): " << deviceProp.major << "." << deviceProp.minor << std::endl;


    
  cudaDeviceGetAttribute(&WARP_SIZE, cudaDevAttrWarpSize, 0); 
  cublasCheck(cublasCreate(&cublas_handle));
  cublasCheck(cublasLtCreate(&cublaslt_handle));


  int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
  printf("enable_tf32: %d\n", enable_tf32);
  
  cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
  // setup the (global) cuBLASLt workspace
  cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));
  
  cudnnCreate(&cudnn);

  std::cout << "Tile size is: " << TILE_SIZE << ".\n\n";


  // Create Global CUDA Streams
  for(int i=0; i<num_parallel_streams; i++)
  {
    CudaStreams *cuda_stream = new CudaStreams();

    cudaStreamCreate(&cuda_stream);
    cuda_stream->idx = i;
    parallel_streams[i] = cuda_stream;

    open_streams[i]=1;
  }

  
  // Set the Main Stream
  main_stream = AllocateStream();
  cublasSetStream(cublas_handle, main_stream);
  cudnnSetStream(cudnn, main_stream);
  ThreadsStream[0] = main_stream;



  if (pthread_mutex_init(&mutex, NULL) != 0) {
    printf("Mutex initialization failed\n");
    return 1;
  }
  if (pthread_mutex_init(&clean_scope_mutex, NULL) != 0) {
    printf("Mutex initialization failed\n");
    return 1;
  }
  if (pthread_mutex_init(&char_pool_mutex, NULL) != 0) {
    printf("Mutex initialization failed\n");
    return 1;
  }
  if (pthread_mutex_init(&vocab_mutex, NULL) != 0) {
    printf("Mutex initialization failed\n");
    return 1;
  }
  pthread_mutex_init(&random_seed_mutex, NULL);
  pthread_mutex_init(&aux_mutex, NULL);
  
  


  lockVars["mutex"] = &mutex;
  



  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter(); // Prepare for target hardware
  InitializeNativeTargetAsmParser();

  leaf_ops = {leaf, tensor_leaf, weight_leaf, bias_leaf};
  activation_ops = {relu_op, gelu_op, softmax_op, tanh_op, sigmoid_op, cudnn_relu_op};
  loss_ops = {cross_entropy_op, cross_entropy_idx_op, mse_op, mse_is_w_op};

  custom_ops = {sigmoid_add2weights_op, embedding_op};

  tensor_scalar_ops = {scalar_add_op, scalar_sub_op, scalar_mult_op, scalar_div_op};

  weightless_ops = {add_op, lgrad_op, dropout_op};
  weightless_ops = concat_int_vec(weightless_ops, tensor_scalar_ops);

  preprocessing_ops = {gpu_op, crop_op, random_horizontal_flip_op, normalize_img_op, jitter_op};
  gradless_ops = {randu_like_op, onehot_op, max_op, argmax_op, equal_op,
                  create_tensor_from_brackets_op, detach_op};
  gradless_ops = concat_int_vec(gradless_ops, preprocessing_ops);


  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence[tok_space] = 1;
  BinopPrecedence['='] = 4;
  BinopPrecedence[':'] = 9;
  BinopPrecedence['!'] = 9;
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
  BinopPrecedence['^'] = 50;
  BinopPrecedence['@'] = 60;


  floatFunctions["log"] = "logE";
  floatFunctions["log2"] = "logE2";
  floatFunctions["log2f"] = "logE2f";
  floatFunctions["round"] = "roundE";
  floatFunctions["floor"] = "floorE";


  stringMethods["split"] = "SplitString";
  stringMethods["split_idx"] = "SplitStringIndexate";




  return_tensor_functions = {"gelu", "sigmoid", "_tanh", "relu", "softmax", "log", "randu_like",
                             "RandomCrop", "RandomHorizontalFlip", "NormalizeImg", "dropout", "sigmoid_add2weights",
                             "rl_discounted_return", "self_attn", "Jitter", "mse_with_priorities",
                             "btc_mult", "btc_multT"};

  return_tensor_methods = {"view", "clip", "argmax", "tmax", "onehot", "shape", "permute", "cpu", "printtt",
                            "sum", "prod", "mean", "tmin", "argmin", "topk", "repeat_interleave",
                            "save_img", "gpu", "gpuw", "save_as_int", "save_as_bin", "gather"};
  
  

  return_tensor_fn = concat_str_vec(return_tensor_functions, return_tensor_methods);

  return_pinned_methods = {"gpu", "gpuw"};


  // Universal
  vararg_methods = {"view", "sum", "mean", "prod", "tmax", "argmax", "load_bin_idx"};
  string_methods = {"split", "split_idx"};


  // tensor + string + ...
  // e.g: x.view(), str.split()
  native_methods = {"split", "split_idx", "first_nonzero", "append"};
  native_methods = concat_str_vec(native_methods, return_tensor_methods);
  //native_methods = concat_str_vec(native_methods, return_pinned_methods);

  return_string_fn = {"to_string", "cat_str_float"};

  require_scope_functions = {"network_ema"};

  native_functions = {"ShuffleStrVec", "gload_img", "wload_img", "silent_sleep", "sleep",
                      "LenStrVec", "zeros_vec", "ones_vec", "start_timer", "end_timer",
                      "_glob_b_", "print", "cross_entropy", "backprop", "AdamW", "SGD",
                      "load_preprocess_img", "max", "min", "unbug", "is_null",
                      "cpu_idx", "eval", "train", "OneCycleLR", "CosineLR", "wload_img_resize",
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


  native_modules = {"ConvForward2d", "MaxPoolForward2d", "BatchNormForward2d", "BN2dReluForward",
                    "ReluForward", "LSTMForward", "EmbeddingForward", "MHSAForward", "LinearForward"};

  threaded_tensor_functions = {"log2", "network_ema", "priority_sample", "priority_sample_val", "importance_sample_idx", "importance_sample_weight"};
  threaded_tensor_functions = concat_str_vec(threaded_tensor_functions, native_modules);
  threaded_tensor_functions = concat_str_vec(threaded_tensor_functions, return_tensor_functions);
  threaded_tensor_functions = concat_str_vec(threaded_tensor_functions, return_tensor_methods);



  tensor_inits = {"binary", "arange", "int", "randu", "zeros", "ones", "xavu", "xavu_relu", "xavu_tanh", "he_normal_relu", "init_gpt", "xavn", "normal"};
  notators_str = {"bias", "fp32", "fp16", "causal"};


  // Prime the first token.
  //fprintf(stderr, "ready> ");
  getNextToken();

  TheJIT = ExitOnErr(KaleidoscopeJIT::Create());
  InitializeModule();

  // Run the main "interpreter loop" now.
  

  MainLoop();

  return 0;
}