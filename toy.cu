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
#include <glob.h>
#include <chrono>
#include <thread>
#include <random>
#include <float.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>



// Cuda
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>
// #include <cute/tensor.hpp>


// #include "include/cutlass/gemm/device/gemm.h"
// #include "include/cutlass/cutlass.h"

#include "src/include.h"



float TERMINATE_VARARG = -40370000000.0f;


#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16


cudaDeviceProp deviceProp;

int WARP_SIZE;

int THREADS_PER_BLOCK = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

const int TILE_SIZE = (int)floorf(sqrtf((float)THREADS_PER_BLOCK)); 
const int TILE_SIZE_SQ = TILE_SIZE*TILE_SIZE;








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


int ASYNC_LOADER_THREADS = 6;
const int num_parallel_streams=32; //global of src/mma/tensor_struc.h
CudaStreams *parallel_streams[num_parallel_streams];
cudaEvent_t parallel_events[num_parallel_streams];
std::vector<cudaEvent_t> Registered_Events;
int open_streams[num_parallel_streams];
CudaStreams *main_stream, *backward_stream;
std::map<int, cudaStream_t> ThreadsStream;
std::vector<int> leaf_ops, loss_ops, gradless_ops, activation_ops, preprocessing_ops, tensor_scalar_ops, custom_ops, weightless_ops;
int nn_mode=training_mode;

std::map<std::string, int> NotatorsMap = {
  {"bias", bias},
  {"fp32", fp32},
  {"fp16", fp16},
  {"causal", causal},
};

bool ShallCodegen = true;


// Tensors
std::map<std::string, Tensor *> NamedTensorsT;
std::map<std::string, float *> NamedPinnedTensors;
std::map<std::string, std::vector<float>> NamedDims;
std::vector<Tensor> TensorsToDelete;


LCG rng(generate_custom_seed());



std::vector<std::string> rds;


pthread_mutex_t mutex, clean_scope_mutex, char_pool_mutex, vocab_mutex, random_seed_mutex, aux_mutex;

  // Error Colors
// \033[0m default
// \033[31m red
// \033[33m yellow
// \033[95m purple






// Tensor related
std::vector<std::string> return_tensor_functions, return_tensor_methods, return_tensor_fn, native_modules,
return_pinned_methods, vararg_methods, string_methods, native_methods, native_functions, native_fn, tensor_inits,
return_string_fn, threaded_tensor_functions, require_scope_functions, notators_str;










//global
std::vector<std::string> objectVars;
std::vector<std::string> globalVars;
std::map<std::string, std::string> functionVars;
std::map<std::string, std::string> floatFunctions;
std::map<std::string, std::string> stringMethods;
std::map<std::string, pthread_mutex_t *> lockVars;



//global
std::vector<std::string> Classes;
std::map<std::string, std::string> Object_toClass;
std::map<std::string, std::string> Object_toClassVec;



std::map<size_t, std::vector<char *>> CharPool;



static std::map<std::string, std::string> objectVecs;
static std::map<std::string, int> objectVecsLastId;






//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//

//global


std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
ExitOnError ExitOnErr;


// Vars
std::map<std::string, Value *> NamedValues;
std::map<std::string, char *> NamedStrs;
std::map<std::string, AllocaInst *> NamedStrVecs;
std::map<std::string, std::vector<char *>> ClassStrVecs;
std::map<std::string, std::vector<float>> ClassFloatVecs;
std::map<std::string, float> NamedClassValues;
std::map<std::string, std::string> NamedObjects;
std::map<std::string, std::vector<std::pair<std::string, std::string>>> ScopeVarsToClean;
std::map<std::string, char *> ScopeNamesToClean;
std::map<int, std::map<std::string, std::vector<std::string>>> ThreadedScopeTensorsToClean;


// Aux to not lose pointers
std::map<std::string, std::vector<char *>> StrVecAuxHash;



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






extern "C" float Add(float value, float v2)
{
  return value + v2; 
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








extern "C" float printtt(int thread_id, Tensor tensor)
{
  char* tensorName = new char[tensor.name.size() + 1]; // Allocate memory for the C-style string
  std::strcpy(tensorName, tensor.name.c_str()); // Copy the string

  PrintTensor(thread_id, tensorName);

  delete[] tensorName;
  return 0;
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

  gelu_backward1<<<grid_size, block_size, 0, main_stream->stream>>>(dinp, inp, dout, dims_prod);
  
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

  sigmoid_backward_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(dinp, out, dout, dims_prod);
  
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
  sigmoid_add2weights_kernel<<<grid_size, block_size, shared_mem_size, main_stream->stream>>>
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
  

  tanh_backward_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(dinp, out, dout, dims_prod);
  
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

  relu_backward1<<<grid_size, block_size, 0, main_stream->stream>>>(inp, dinp, dout, dims_prod);
  
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


  gather_last_dim_backward_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(dx, dy, idx, leading_dim, dims_prod);

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
      // w_cpu = make_normal(OC*C);
      w_cpu = make_embedding_uniform(OC*C);


      
      W = get_from_pool(0, OC*C, "Embedding W");
      cudaMemcpy(W, w_cpu, OC*C*sizeof(float), cudaMemcpyHostToDevice);

      Tensor *tensor_W = createTensor(W, {(float)C,(float)OC}, OC*C, true, Name);
      
      
      


      dW = get_from_pool(0, OC*C, "embedding dW");
      set_to_zero_kernel<<<std::ceil((OC*C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream->stream>>>(dW, OC*C);

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








template<int tile_size, int block_rows>
__global__ void transpose_kernel(const float *__restrict__ X, float *__restrict__ Y)
{
  __shared__ float tile[tile_size * tile_size];

  int x = blockIdx.x * tile_size + threadIdx.x;
  int y = blockIdx.y * tile_size + threadIdx.y;
  int width = gridDim.x * tile_size;

  for (int j = 0; j < tile_size; j += block_rows)
     tile[(threadIdx.y+j)*tile_size + threadIdx.x] = X[(y+j)*width + x];

  __syncthreads();

  for (int j = 0; j < tile_size; j += block_rows)
     Y[(y+j)*width + x] = tile[(threadIdx.y+j)*tile_size + threadIdx.x];          
}



inline void transpose(Tensor *tensor, int thread_id, cudaStream_t stream)
{

  float *transposed = get_from_pool(thread_id, tensor->dims_prod, "transpose");


  constexpr int tile_size{32}; // todo

  dim3 grid_size(std::ceil(tensor->dims[0]/(float)tile_size), std::ceil(tensor->dims[1]/(float)tile_size));
  dim3 block_size(tile_size, 8);

  transpose_kernel<tile_size, 8><<<grid_size, block_size, 0, stream>>>(tensor->tensor_ptr, transposed);

  // move_to_pool(thread_id, tensor->dims_prod, tensor->tensor_ptr, "transpose");
  tensor->tensor_ptr = transposed;
}


__global__ void float_to_half_kernel(const float *__restrict__ x, __half *__restrict__ y, const int dims_prod)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx>dims_prod)
    return;

  y[idx] = __float2half(x[idx]);
}


inline Tensor *float_to_half(Tensor *tensor, int thread_id, cudaStream_t stream)
{

  half *tensor_ptr = get_half_from_pool(thread_id, tensor->dims_prod, "float to half");


  Tensor *half_tensor = createTensorHalf(tensor_ptr, tensor->dims, tensor->dims_prod, true, tensor->name+"half");

  float_to_half_kernel<<<std::ceil(tensor->dims_prod/(float)THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(tensor->tensor_ptr, tensor_ptr, tensor->dims_prod);

  return half_tensor;
}
inline half *float_to_half(float *tensor, int thread_id, int dims_prod, cudaStream_t stream)
{
  half *tensor_ptr = get_half_from_pool(thread_id, dims_prod, "float to half");


  float_to_half_kernel<<<std::ceil(dims_prod/(float)THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(tensor, tensor_ptr, dims_prod);

  return tensor_ptr;
}





__global__ void half_to_float_kernel(const __half *__restrict__ x, float *__restrict__ y, const int dims_prod)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx>dims_prod)
    return;

  y[idx] = __half2float(x[idx]);
}



inline Tensor *half_to_float(Tensor *tensor, int thread_id, cudaStream_t stream)
{

  float *tensor_ptr = get_from_pool(thread_id, tensor->dims_prod, "half to float");
  


  Tensor *float_tensor = createTensor(tensor_ptr, tensor->dims, tensor->dims_prod, true, tensor->name+"half");

  half_to_float_kernel<<<std::ceil(tensor->dims_prod/(float)THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(tensor->half_ptr, tensor_ptr, tensor->dims_prod);

  return float_tensor;
}
inline float *half_to_float(half *tensor, int thread_id, int dims_prod, cudaStream_t stream)
{

  float *tensor_ptr = get_from_pool(thread_id, dims_prod, "half to float");
  

  half_to_float_kernel<<<std::ceil(dims_prod/(float)THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(tensor, tensor_ptr, dims_prod);

  return tensor_ptr;
}
inline float *half_to_float_overwrite(half *tensor, float *float_tensor, int dims_prod, cudaStream_t stream)
{
  half_to_float_kernel<<<std::ceil(dims_prod/(float)THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(tensor, float_tensor, dims_prod);

  return float_tensor;
}




float *MHSA::Forward(Tensor *x, int B, int T, int thread_id)
{
  //std::cout << "MHSA::Forward" << "\n";


  

  float *proj_out = get_from_pool(thread_id, B*T*C, "mhsa out");

  cudaStream_t stream = ThreadsStream[thread_id];  

  if (this->B!=B || this->T!=T)
    SetDescriptors(B, T, thread_id);



  
  //std::cout << "" << main_stream->stream==stream << "\n";

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
    dim3 block_size_mhsa(8, WARP_SIZE);

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

    // constexpr int num_warps_x{4};
    // constexpr int num_warps_y{4};
    

    constexpr int WMMA_T{16};
    // dim3 block_size_wmma(num_warps_x * WARP_SIZE, num_warps_y);
    // dim3 grid_size_wmma_proj(std::floor((3*C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::floor((B*T + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    // int shared_mem_wmma = (num_warps_y*WMMA_T*WMMA_T*num_warps_x+ WMMA_T*WMMA_T)*sizeof(float) + (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);
    // wmma_mult_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_wmma_proj, block_size_wmma, shared_mem_wmma, stream>>>(x->tensor_ptr, W, qkv, B*T, C, 3*C);

    // int shared_mem_cf = (num_warps_y*WMMA_T*WMMA_T*num_warps_x)*sizeof(float);
    // wmma_cp_async<WMMA_T,num_warps_x,num_warps_y><<<grid_size_wmma_proj, block_size_wmma, shared_mem_cf, stream>>>(x->tensor_ptr, W, qkv, B*T, C, 3*C);


    blocking_mma<WMMA_T>(x->tensor_ptr, W, qkv, B*T, C, 3*C, stream);



    // Tensor *x_half = float_to_half(x, thread_id, stream);
    // Tensor *w_half = createTensor(W, {(float)3*(float)C*(float)C}, 3*C*C, true, "w transpose");
    // // transpose(w_half, thread_id, stream);
    // w_half = float_to_half(w_half, thread_id, stream);
    // half *qkv_half = get_half_from_pool(thread_id, B*T*3*C, "qkv half");
    // mmaAsyncStage3(x_half->half_ptr, w_half->half_ptr, qkv_half, B*T, C, 3*C);
    // qkv = half_to_float_overwrite(qkv_half, qkv, B*T*3*C, stream);





    // cudaCheck(cudaGetLastError());
    
    
    //cudaStreamSynchronize(stream);
    //PrintTensorF(qkv, B*T, 3*C);

    // Attention

    constexpr int num_warps{8};

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

    blocking_mma<WMMA_T>(out, W_proj, proj_out, B*T, C, C, stream);
                          
    // cudaCheck(cudaGetLastError());
    
    // move_to_pool(thread_id, B*T*C, x_half->half_ptr, "MHSA qkv");
    // move_to_pool(thread_id, 3*C*C, w_half->half_ptr, "MHSA qkv");
    // move_to_pool(thread_id, B*T*3*C, qkv_half, "MHSA qkv");
    // // delete[] x_half;
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

  set_to_zero_kernel<<<std::ceil((3*C*C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream->stream>>>(dW, 3*C*C);
  set_to_zero_kernel<<<std::ceil((C*C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream->stream>>>(dW_proj, C*C);

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


    
    //StreamAwaitStreamB(dw_proj_stream, main_stream->stream);
    


    //mult_backwarddw<<<grid_size_dwproj, block_size, shared_mem_size, main_stream->stream>>>(out, dW_proj, dy, TILE_SIZE, TILE_SIZE_SQ, B*T, C, C);
    //mult_backwarddx<<<grid_size_dwproj, block_size, shared_mem_size, main_stream->stream>>>(W_proj, d_out, dy, TILE_SIZE, TILE_SIZE_SQ, B*T, C, C);


    
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, C, B*T, &one,
                              out, CUBLAS_LOWP, C, dy, CUBLAS_LOWP, C, &one,
                              dW_proj, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, C, &one,
                              W_proj, CUBLAS_LOWP, C, dy, CUBLAS_LOWP, C, &zero,
                              d_out, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    



    //PrintTensorF(d_out, T, C);


    dim3 grid_size_mhsa(nh, B);
    dim3 block_size_mhsa(8, WARP_SIZE);

    int threads_per_block = block_size_mhsa.x*block_size_mhsa.y;
    int warps_per_block = threads_per_block/WARP_SIZE;


    //int last_id = ((4*Bc_back + 4*Br_back)*d + 3*Br_back*Bc_back + 2*Br_back)*sizeof(float);
    //std::cout << "backward last_id: " << last_id << ", M: " << M << "\n";
    //std::cout << "Bc: " << Bc_back << ", Br: " << Br_back << ", Tc: " << Tc_back << ", Tr: " << Tr_back << "\n";
    flash_attn_backward_kernel<<<grid_size_mhsa, block_size_mhsa, M, main_stream->stream>>>(d_qkv, d_out, qkv, out, l, D,
                                                                                B, nh, T, d, C, sqrtf(d),
                                                                                Bc_back, Br_back, Tc_back, Tr_back,
                                                                                warps_per_block, threads_per_block);

    //PrintTensorF(d_qkv, 3, C);
    //PrintTensorF(d_qkv, T, 3*C);
    
    //StreamAwaitStreamB(dw_stream, main_stream->stream);

    dim3 grid_size_dx(std::ceil(C/(float)TILE_SIZE), std::ceil((B*T)/(float)TILE_SIZE));
    dim3 grid_size_dw(std::ceil(C/(float)TILE_SIZE), std::ceil((3*C)/(float)TILE_SIZE));
    //mult_backwarddw<<<grid_size_dw, block_size, shared_mem_size, main_stream->stream>>>(x, dW, d_qkv, TILE_SIZE, TILE_SIZE_SQ, B*T, C, 3*C);
    //mult_backwarddx<<<grid_size_dx, block_size, shared_mem_size, main_stream->stream>>>(W, dx, d_qkv, TILE_SIZE, TILE_SIZE_SQ, B*T, C, 3*C);
    
    
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, 3*C, B*T, &one,
                              x, CUBLAS_LOWP, C, d_qkv, CUBLAS_LOWP, 3*C, &one,
                              dW, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, 3*C, &one,
                              W, CUBLAS_LOWP, C, d_qkv, CUBLAS_LOWP, 3*C, &zero,
                              dx, CUBLAS_LOWP, C, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    

    //StreamAwaitStreamB(main_stream->stream, dw_proj_stream);
    //StreamAwaitStreamB(main_stream->stream, dw_stream);
    //cudaStreamDestroy(dw_proj_stream);
    //cudaStreamDestroy(dw_stream);
  } else {
    

    //cudaStream_t dw_proj_stream, dw_stream;
    //cudaStreamCreate(&dw_proj_stream);
    //cudaStreamCreate(&dw_stream);


    
    //StreamAwaitStreamB(dw_proj_stream, main_stream->stream);
    
    constexpr int num_warps_x{4};
    constexpr int num_warps_y{4};
    
    constexpr int WMMA_T{16};
    
    int shared_mem_size = num_warps_y*WMMA_T*WMMA_T*num_warps_x*sizeof(float) + (num_warps_x+num_warps_y)*WMMA_T*WMMA_T*sizeof(__half);

    dim3 block_size(num_warps_x * WARP_SIZE, num_warps_y);

    dim3 grid_size_dx_proj(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((B*T + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    dim3 grid_size_dw_proj(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((C + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));

    wmma_backwarddx_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dx_proj, block_size, shared_mem_size, main_stream->stream>>>(d_out, W_proj, dy, B*T, C, C);
    wmma_backwarddw_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dw_proj, block_size, shared_mem_size, main_stream->stream>>>(dW_proj, out, dy, B*T, C, C);
    
    


    dim3 grid_size_mhsa(nh, B);
    dim3 block_size_mhsa(8, WARP_SIZE);

    int threads_per_block = block_size_mhsa.x*block_size_mhsa.y;
    int warps_per_block = threads_per_block/WARP_SIZE;


    //int last_id = ((4*Bc_back + 4*Br_back)*d + 3*Br_back*Bc_back + 2*Br_back)*sizeof(float);
    //std::cout << "backward last_id: " << last_id << ", M: " << M << "\n";
    //std::cout << "Bc: " << Bc_back << ", Br: " << Br_back << ", Tc: " << Tc_back << ", Tr: " << Tr_back << "\n";
    flash_attn_backward_kernel<<<grid_size_mhsa, block_size_mhsa, M, main_stream->stream>>>(d_qkv, d_out, qkv, out, l, D,
                                                                                B, nh, T, d, C, sqrtf(d),
                                                                                Bc_back, Br_back, Tc_back, Tr_back,
                                                                                warps_per_block, threads_per_block);

    //PrintTensorF(d_qkv, 3, C);
    //PrintTensorF(d_qkv, T, 3*C);
    
    //StreamAwaitStreamB(dw_stream, main_stream->stream);



    dim3 grid_size_dx(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((B*T + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));
    dim3 grid_size_dw(std::ceil((C + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((3*C + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));

    wmma_backwarddx_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dx, block_size, shared_mem_size, main_stream->stream>>>(dx, W, d_qkv, B*T, C, 3*C);
    wmma_backwarddw_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dw, block_size, shared_mem_size, main_stream->stream>>>(dW, x, d_qkv, B*T, C, 3*C);
    
    
  

    //StreamAwaitStreamB(main_stream->stream, dw_proj_stream);
    //StreamAwaitStreamB(main_stream->stream, dw_stream);
    //cudaStreamDestroy(dw_proj_stream);
    //cudaStreamDestroy(dw_stream);

  }







  //add_forward<<<std::ceil((B*T*C)/THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, main_stream->stream>>>(dx, dx, dy, B*T*C);


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


    
    blocking_mma<WMMA_T>(x->tensor_ptr, W, out, B, C, OC, stream);
    



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

  set_to_zero_kernel<<<std::ceil((OC*C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream->stream>>>(dW, OC*C);

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

    // wmma_dx_cp_async<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dx, block_size, shared_mem_cf, main_stream->stream>>>(dx, W, dy, B, C, OC);
    wmma_backwarddx_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dx, block_size, shared_mem_size, main_stream->stream>>>(dx, W, dy, B, C, OC);
    wmma_backwarddw_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size_dw, block_size, shared_mem_size, main_stream->stream>>>(dW, x, dy, B, C, OC);




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


  //set_to_zero_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(all_ht, B*T*OC);


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
  //cudaStream_t stream = main_stream->stream;





  

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

  set_to_zero_kernel<<<std::ceil((4*OC*OC)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream->stream>>>(dW, 4*OC*OC);
  set_to_zero_kernel<<<std::ceil((4*OC* C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream->stream>>>(dU, 4*OC*C);
  set_to_zero_kernel<<<std::ceil((4*OC)   /(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream->stream>>>(dB, 4*OC);

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
  copy_tensor_kernel<<<std::ceil(((float)B*(float)OC)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream->stream>>>(d_ht, dy, B*OC);
  set_to_zero_kernel<<<std::ceil(((float)B*(float)OC)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream->stream>>>(d_ct,     B*OC); // TODO: check if removing this one is safe
  set_to_zero_kernel<<<std::ceil(((float)T*(float)B*4*(float)OC)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream->stream>>>(d_ifoc, T*B*4*OC);


  

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

    lstm_elementwise_ops_backward_kernel<<<grid_size_elementwises, block_size, shared_mem_size, main_stream->stream>>>(fused_out,
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
    lstm_single_step_backward_dht_kernel<<<grid_size_d_ht, block_size, shared_mem_size, main_stream->stream>>>(d_ifoc,
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

  cudaStreamSynchronize(main_stream->stream);

  
  lstm_backward_dx_kernel<<<grid_size_dx, block_size, shared_mem_size, dx_stream>>>(d_ifoc, dx, U,
                                                                                      TILE_SIZE, TILE_SIZE_SQ, B, T, C, 4*OC);
  RegisterEvent(dx_stream);
  
  

  
  mult_backwarddw<<<grid_size_dw, block_size, shared_mem_size, dw_stream>>>(all_ht, dW, d_ifoc, TILE_SIZE, TILE_SIZE_SQ, B*T, OC, 4*OC);
  RegisterEvent(dw_stream);

  
  mult_backwarddw<<<grid_size_du, block_size, shared_mem_size, main_stream->stream>>>(x, dU, d_ifoc, TILE_SIZE, TILE_SIZE_SQ, B*T, C, 4*OC);

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
  set_to_zero_kernel<<<std::ceil((B*OC)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream->stream>>>(dW, B*OC);

  changed_descriptors=false;
}

void Embedding::Backward(float *x, float *dy)
{
  /*
  if(changed_descriptors)
    SetBackwardDescriptors();
  //dW = dy;
  copy_tensor_kernel<<<std::ceil((B*OC)/(float)THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, main_stream->stream>>>(dW, dy, B*C);
  */

  

  

  dim3 block_size(TILE_SIZE, TILE_SIZE);
  dim3 grid_size(std::ceil(OC/(float)TILE_SIZE), std::ceil(B/(float)TILE_SIZE));
  embedding_backward_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(x, dW, dy, TILE_SIZE, B, C, OC);
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


  random_padding_cropping_kernel<<<numBlocks, threadsPerBlock, 0, tensor->cuda_stream->stream>>>(
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

  random_horizontal_flip_kernel<<<numBlocks, threadsPerBlock, 0, tensor->cuda_stream->stream>>>(
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

  normalize_img_kernel<<<numBlocks, threadsPerBlock, 0, tensor->cuda_stream->stream>>>(
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

  scalarmult_backward_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(dx, dy, scalar, dims_prod);
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

  hadamard_backward_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(x, w, dx, dw, dy, dims_prod);
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

    dropout_mask_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(device_y, dropout_ptr, tensor->tensor_ptr, rate, scale, dims_prod, seed);
    
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

  dropout_backward_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(dx, mask, dy, dims_prod);
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

  set_to_zero_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(probs, B*C);


  /*
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*32*C);
  grid_size  = B*32;
  block_size = grid_block_mem_sizes[1];
  
  online_softmax<<<grid_size, block_size, 0, main_stream->stream>>>(y_hat, probs, B, C);
  */
  
  
  

  
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size  = B;
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = 2 * block_size / 32 * sizeof(float);

  softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size, main_stream->stream>>>(y_hat, probs, B, C);
  
  


  
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  
  crossentropy_softmax_backward_kernel1<<<grid_size, block_size, 0, main_stream->stream>>>(dloss, probs, y, B, C, scale);
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

  set_to_zero_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(probs, B*C);


  

  /*
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size  = B;
  block_size = grid_block_mem_sizes[1];
  shared_mem_size = 2 * block_size / 32 * sizeof(float);

  softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size, main_stream->stream>>>(y_hat, probs, B, C);
  */
  grid_block_mem_sizes = CalculateSimpleWarpGridAndBlockSizes(B);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  online_softmax<<<grid_size, block_size, 0, main_stream->stream>>>(y_hat, probs, B, C);
  
  


  
  grid_block_mem_sizes = CalculateGridAndBlockSizes(B*C);
  grid_size = grid_block_mem_sizes[0];
  block_size = grid_block_mem_sizes[1];

  
  crossentropy_idx_backward_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(dloss, probs, y, B, C, scale);
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
  
  mse_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(dloss, y_hat, y, scale, dims_prod);

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
  

  online_mse<<<grid_size, block_size, 0, main_stream->stream>>>(msed, y_hat->tensor_ptr, y->tensor_ptr, B, C);

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
  mse_with_priorities_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(dloss, y_hat_tensor->tensor_ptr, y_tensor->tensor_ptr, is_w_tensor->tensor_ptr, scale, dims_prod);

  //PrintTensorF(dloss, 1, dims_prod);
}


//




//




//











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
            set_to_zero_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(new_grad_ptr, w_size);
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
            set_to_zero_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(new_grad_ptr, b_size);
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
          set_to_zero_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(device_dw, w_size);
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
      set_to_zero_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(device_dx, x_size);
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
   
  //  float grad = std::clamp(grads_memory[i], -grad_clip, grad_clip) + weight_decay * params_memory[i];
   float grad = grads_memory[i];
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
   
  //  float grad = std::clamp(grads_memory[i], -grad_clip, grad_clip);
   float grad = grads_memory[i];
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
    //StreamAwaitStreamB(streams[i], main_stream->stream);
  }

  cudaStreamSynchronize(main_stream->stream);

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
    //StreamAwaitStreamB(main_stream->stream, streams[i]);
  }
  for (int i = 0; i < num_streams; ++i)
    cudaStreamDestroy(streams[i]);

  cudaStreamSynchronize(main_stream->stream);

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

      std::cout << "ATTRIBUTING TO STRING VEC: " << Lname << "\n";

    } else if (LType=="float_vec") {

      //std::cout << "ATTRIBUTING TO FLOAT VEC: " << Lname << ", type: " << Type << ", is vec: " << LHS->GetIsVec() << "\n";

      

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
      
      Builder->CreateCall(TheModule->getFunction("StoreStrVecOnDemand"),
                                                  {Lvar_name,
                                                   Val});

    } else if (LType=="float_vec") {

      //std::cout << "ATTRIBUTING TO FLOAT VEC: " << Lname << ", type: " << Type << ", is vec: " << LHS->GetIsVec() << "\n";

      

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





Function *codegenAsyncFunction(std::vector<std::unique_ptr<ExprAST>> &asyncBody, Value *first_arg, Value *scope_str, Value *previous_scope, Value *thread_id, Value *has_grad) {
  

  // find existing unique function name (_async_1, _async_2, _async_3 etc)
  int fnIndex = 1;
  while (TheModule->getFunction("__async_" + std::to_string(fnIndex)))
    fnIndex++;
  
  CudaStreams *thread_stream = AllocateStream(0);
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

      //var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
      //                                        {object_hash, var_name});


      Builder->CreateCall(TheModule->getFunction("InitObjectVecWithNull"),
                                                {var_name, init});
    } else
    {}
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
    
    std::string type = "";
    if (typeVars.find(arg_name) != typeVars.end())
      type = typeVars[arg_name];

    // Coder args    
    if (type=="float")
    {
      Value *var_name = Builder->CreateGlobalString(arg_name);
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"),
                                    {scope_str, var_name});

      Builder->CreateCall(TheModule->getFunction("StoreArgOnDemand"),
                                                  {scope_str, var_name, &Arg});
    } else if (type=="str")
    {
      Value *var_name = Builder->CreateGlobalString(arg_name);
      var_name = Builder->CreateCall(TheModule->getFunction("ConcatStr"), //TODO: Store scope vars to clean for this too
                                    {scope_str, var_name});


      Builder->CreateCall(TheModule->getFunction("StoreStrOnDemand"),
                                                  {var_name, &Arg});
    }
    else if (type!="tensor")
    {
      AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, Arg.getName());
      Builder->CreateStore(&Arg, Alloca);
      //Builder->CreateStore(Builder->CreateLoad(Type::getFloatTy(*TheContext), &Arg), Alloca);



      NamedValues[std::string(Arg.getName())] = Alloca;

    }
    else
    {
      if (type=="tensor")
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
  TheModule->getOrInsertFunction("__slee_p_", sleepTy);

  
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
  



  FunctionType *Add_String_To_NotesVector = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("Add_String_To_NotesVector", Add_String_To_NotesVector);

  FunctionType *Add_Float_To_NotesVector = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getFloatTy(*TheContext)},
      false 
  );
  TheModule->getOrInsertFunction("Add_Float_To_NotesVector", Add_Float_To_NotesVector);


  FunctionType *CreateNotesVector = FunctionType::get(
      int8PtrTy,
      {},
      false 
  );
  TheModule->getOrInsertFunction("CreateNotesVector", CreateNotesVector);

  FunctionType *Dispose_NotesVector = FunctionType::get(
    Type::getFloatTy(*TheContext),
    {int8PtrTy},
    false 
  );
  TheModule->getOrInsertFunction("Dispose_NotesVector", Dispose_NotesVector);


  FunctionType *str_Create = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("str_Create", str_Create);

  FunctionType *float_Create = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("float_Create", float_Create);


  FunctionType *str_vec_Create = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("str_vec_Create", str_vec_Create);

  FunctionType *float_vec_Create = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("float_vec_Create", float_vec_Create);
  
  FunctionType *tensor_Create = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("tensor_Create", tensor_Create);

  FunctionType *pinned_tensor_Create = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext), int8PtrTy, Type::getInt32Ty(*TheContext), int8PtrTy},
      false 
  );
  TheModule->getOrInsertFunction("pinned_tensor_Create", pinned_tensor_Create);


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

__attribute__((constructor))
void early_init() {
    // std::cout << "Constructor Function Executed\n";
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter(); // Prepare for target hardware
  InitializeNativeTargetAsmParser();
}

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

    cudaStreamCreate(&cuda_stream->stream);
    cuda_stream->idx = i;
    parallel_streams[i] = cuda_stream;

    open_streams[i]=1;
  }

  
  // Set the Main Stream
  main_stream = AllocateStream(0);
  cublasSetStream(cublas_handle, main_stream->stream);
  cudnnSetStream(cudnn, main_stream->stream);
  ThreadsStream[0] = main_stream->stream;



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

  native_functions = {"ShuffleStrVec", "gload_img", "wload_img", "silent_sleep", "__slee_p_",
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