
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>

#include <string>
#include <vector>


#include "../../../backprop/include.h"
#include "../../../common/cu_commons.h"
#include "../../../cuda_kernels/handles.h"
#include "../../../cuda_kernels/elementwise_kernels_inline.cu"
#include "../../../tensor/include.h"
#include "class.h"
#include "kernels.h"



Embedding::Embedding(int C, int OC, std::string Init, std::string Name)
    : C(C), OC(OC), Init(Init), Name(Name) {
    // C == num_codebooks
    B = 0;

    float *w_cpu;
        
    //w_cpu = make_xavier_uniform_float(OC*C, OC,  C);
    // w_cpu = make_normal(OC*C);
    w_cpu = make_embedding_uniform(OC*C);


    
    W = get_from_pool(0, OC*C, "Embedding W");
    cudaMemcpy(W, w_cpu, OC*C*sizeof(float), cudaMemcpyHostToDevice);

    DT_tensor *tensor_W = createTensor(W, {(float)C,(float)OC}, OC*C, true, Name);
    
    
    


    dW = get_from_pool(0, OC*C, "embedding dW");
    set_to_zero_kernel<<<std::ceil((OC*C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dW, OC*C);

    NamedTensorsT[Name] = tensor_W;
    NamedParamGrads[Name] = dW;

    delete[] w_cpu;

    changed_descriptors=false;
}


void Embedding::SetDescriptors(int B)
{
  this->B=B;
  changed_descriptors=true;
}


float *Embedding::Forward(DT_tensor *tensor, int B, int thread_id)
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