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

  std::cout << "a" << ".\n";

  PrintTensor(thread_id, tensorName);

  delete[] tensorName;
  return 0;
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
  std::cout << "a" << ".\n";

}





















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
      Builder->CreateCall(TheModule->getFunction("float_Store"),
                                                  {Lvar_name,
                                                   Val, thread_id});

    } else if (LType=="str") {


      Builder->CreateCall(TheModule->getFunction("str_Store"),
                                                  {Lvar_name,
                                                   Val, thread_id});
                                                   

    } else if (LType=="str_vec") {

      std::cout << "ATTRIBUTING TO STRING VEC: " << Lname << "\n";

    } else if (LType=="float_vec") {

      //std::cout << "ATTRIBUTING TO FLOAT VEC: " << Lname << ", type: " << Type << ", is vec: " << LHS->GetIsVec() << "\n";

      

      if(LHS->GetIsVec())
      {
        VecIdxExprAST *LHSV = static_cast<VecIdxExprAST *>(LHS.get());
        

        Builder->CreateCall(TheModule->getFunction("float_vec_Store_Idx"),
                                                {Lvar_name,
                                                  LHSV->Idx[0]->codegen(first_arg, scope_str, previous_scope, thread_id, has_grad),
                                                  Val, thread_id});

      } else
        Builder->CreateCall(TheModule->getFunction("float_vec_Store"),
                                                {Lvar_name,
                                                  Val});
        

    } else {
      
      seen_var_attr=false;
      
      
      Builder->CreateCall(TheModule->getFunction("float_Store"),
                                                  {Lvar_name,
                                                   Val, thread_id});
      

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
        

      Value *tensorPtr = Builder->CreateCall(TheModule->getFunction("tensor_Load"),
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
        ArgsV[1] = Builder->CreateCall(TheModule->getFunction("tensor_Load"), {ArgsV[1]});
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

      arg = Builder->CreateCall(TheModule->getFunction("tensor_Load"), {arg});
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
      Value *V = Builder->CreateCall(TheModule->getFunction("str_Load"), 
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


      Builder->CreateCall(TheModule->getFunction("str_Store"),
                                                  {var_name, &Arg, thread_id});
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
  FunctionType *float_vec_StoreTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getFloatTy(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("float_vec_Store", float_vec_StoreTy);

  FunctionType *float_vec_Store_IdxTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("float_vec_Store_Idx", float_vec_Store_IdxTy);


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
  FunctionType *float_StoreTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, Type::getFloatTy(*TheContext), Type::getInt32Ty(*TheContext)},
      false //
  );
  TheModule->getOrInsertFunction("float_Store", float_StoreTy);

  FunctionType *str_StoreTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
      false //
  );
  TheModule->getOrInsertFunction("str_Store", str_StoreTy);

  FunctionType *str_vec_StoreTy = FunctionType::get(
      Type::getVoidTy(*TheContext),
      {int8PtrTy, int8PtrTy, Type::getInt32Ty(*TheContext)},
      false //
  );
  TheModule->getOrInsertFunction("str_vec_Store", str_vec_StoreTy);
  
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
  FunctionType *float_LoadTy = FunctionType::get(
      Type::getFloatTy(*TheContext),
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("float_Load", float_LoadTy);

  FunctionType *float_vec_LoadTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("float_vec_Load", float_vec_LoadTy);

  FunctionType *str_LoadTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("str_Load", str_LoadTy);

  FunctionType *str_vec_LoadTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("str_vec_Load", str_vec_LoadTy);
  
  FunctionType *tensor_LoadTy = FunctionType::get(
      int8PtrTy,
      {int8PtrTy, Type::getInt32Ty(*TheContext)},
      false
  );
  TheModule->getOrInsertFunction("tensor_Load", tensor_LoadTy);

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



  ops_type_return = {{"tensor_tensor", "tensor"}, {"float_float", "float"}, {"str_str", "str"}, {"str_float", "str"},
                     {"tensor_float", "tensor"}, {"pinned_tensor_pinned_tensor", "pinned_tensor"},
                     {"pinned_tensor_tensor", "pinned_tensor"}, {"pinned_tensor_float", "pinned_tensor"},
                     {"object_object", "object"}, {"str_object", "object"}};

  op_map = {{'*', "mult"}, {'+', "sum"}, {'-', "subtract"}, {'/', "divide"}, {'<', "minor"}, {'>', "higher"}, {tok_equal, "equal"},
            {tok_diff, "different"}, {'/', "divide"}, {tok_higher_eq, "higher_eq"}, {tok_minor_eq, "minor_eq"}, {'%', "mod"},
            {77, "error"}};



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