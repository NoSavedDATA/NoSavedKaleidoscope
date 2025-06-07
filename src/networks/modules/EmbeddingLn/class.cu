
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
#include "../../../mma/util.h"
#include "../../../nsk_cuda/pool/include.h"
#include "../../../tensor/include.h"
#include "class.h"
#include "kernels.h"


DT_EmbeddingLn::DT_EmbeddingLn(int V, int C, int OC, std::string Init, std::string Name)
    : V(V), C(C), OC(OC), Init(Init), Name(Name) {
    // C == num_codebooks
    B = 0;

    float *w_cpu, *book_cpu;
        
    //w_cpu = make_xavier_uniform_float(OC*C, OC,  C);
    // w_cpu = make_normal(OC*C);
    book_cpu = make_embedding_uniform(V*C);
    w_cpu = make_xavier_uniform_float(OC*C, C, OC);



    
    
    Book = get_from_pool(0, V*C, "Embedding Book");
    W = get_from_pool(0, OC*C, "Embedding W");

    cudaMemcpy(Book, book_cpu, V*C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W, w_cpu, OC*C*sizeof(float), cudaMemcpyHostToDevice);

    DT_tensor *tensor_Book = createTensor(Book, {V, C}, V*C, true, Name+"_Book");
    DT_tensor *tensor_W = createTensor(W, {OC, C}, OC*C, true, Name+"_W");
    
    
    


    dBook = get_from_pool(0, V*C, "embedding dW");
    dW = get_from_pool(0, OC*C, "embedding dW");
    set_to_zero_kernel<<<std::ceil((V*C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dBook, V*C);
    set_to_zero_kernel<<<std::ceil((OC*C)/(float)TILE_SIZE_SQ), TILE_SIZE_SQ, 0, main_stream>>>(dW, OC*C);

    NamedTensorsT[Name+"_Book"] = tensor_Book;
    NamedParamGrads[Name+"_Book"] = dBook;
    
    NamedTensorsT[Name+"_W"] = tensor_W;
    NamedParamGrads[Name+"_W"] = dW;

    delete[] book_cpu;
    delete[] w_cpu;

    changed_descriptors=false;
}


void DT_EmbeddingLn::SetDescriptors(int B)
{
  this->B=B;
  changed_descriptors=true;
}




template<int wx_per_wmma_m, int wy_per_wmma_n>
void launch_embedding_ln(Wmma_Grid grid, const float *embedding_book, const float *idxs, const float *weight, float *out, const int M, const int N, const int K, cudaStream_t stream) {

  embeddingln_forward_kernel<wx_per_wmma_m, wy_per_wmma_n>
                            <<<grid.g, grid.w, grid.smem, stream>>>(embedding_book, idxs, weight, out,
                                                                    grid.bx_per_w, grid.by_per_w, grid.bx_per_wx,
                                                                    grid.bx, grid.by,
                                                                    grid.wx, grid.wy, 32,
                                                                    M, N, K);
}

float *DT_EmbeddingLn::Forward(DT_tensor *tensor, int B, int thread_id)
{
  float *out = get_from_pool(thread_id, B*OC, "embedding out");


  if (this->B!=B)
    SetDescriptors(B);

 
  Wmma_Grid grid = CalculateBlockingSize(OC, B,
                                         8,
                                         128, 64,
                                         32, 32,
                                         16, 16);

  

  cudaStream_t stream = ThreadsStream[thread_id];

  // using LaunchFn = void(*)(Wmma_Grid, const float*, const float*, const float *, float*, int, int, int, cudaStream_t);
  // static constexpr LaunchFn dispatch_table[5][5] = {
  //     {nullptr}, // 0 is unused
  //     {nullptr, launch_embedding_ln<1,1>, launch_embedding_ln<1,2>, launch_embedding_ln<1,3>, launch_embedding_ln<1,4>},
  //     {nullptr, launch_embedding_ln<2,1>, launch_embedding_ln<2,2>, launch_embedding_ln<2,3>, launch_embedding_ln<2,4>},
  //     {nullptr, launch_embedding_ln<3,1>, launch_embedding_ln<3,2>, launch_embedding_ln<3,3>, launch_embedding_ln<3,4>},
  //     {nullptr, launch_embedding_ln<4,1>, launch_embedding_ln<4,2>, launch_embedding_ln<4,3>, launch_embedding_ln<4,4>},
  // };

  // auto launcher = dispatch_table[grid.wx_per_wmma_m][grid.wy_per_wmma_n];
  // launcher(grid, Book, tensor->tensor_ptr, W, out, B, OC, C, stream);


  embeddingln_forward_kernel<2, 2>
                          <<<grid.g, grid.w, grid.smem, stream>>>(Book, tensor->tensor_ptr, W, out,
                                                                  grid.bx_per_w, grid.by_per_w, grid.bx_per_wx,
                                                                  grid.bx, grid.by,
                                                                  grid.wx, grid.wy, 32,
                                                                  B, OC, C);

  return out;
}



void DT_EmbeddingLn::SetBackwardDescriptors()
{
}

void DT_EmbeddingLn::Backward(float *idxs, float *dy)
{
  /*
  if(changed_descriptors)
    SetBackwardDescriptors();
  //dW = dy;
  copy_tensor_kernel<<<std::ceil((B*OC)/(float)THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, main_stream>>>(dW, dy, B*C);
  */




  
  Wmma_Grid grid_dw = CalculateBlockingSize(C, OC,
                                         8,
                                         128, 64,
                                         32, 32,
                                         16, 16);

  embeddingln_backward_dw<2, 2>
        <<<grid_dw.g, grid_dw.w, grid_dw.smem, main_stream>>>(dy, Book, idxs, dW,
                                                grid_dw.bx_per_w, grid_dw.by_per_w, grid_dw.bx_per_wx,
                                                grid_dw.bx, grid_dw.by,
                                                grid_dw.wx, grid_dw.wy, 32,
                                                OC, C, B);




  Wmma_Grid grid_dx = CalculateBlockingSize(C, B,
                                         8,
                                         128, 64,
                                         32, 32,
                                         16, 16);

  embeddingln_backward_dx<2, 2>
        <<<grid_dx.g, grid_dx.w, grid_dx.smem, main_stream>>>(dy, W, dBook, idxs,
                                                grid_dx.bx_per_w, grid_dx.by_per_w, grid_dx.bx_per_wx,
                                                grid_dx.bx, grid_dx.by,
                                                grid_dx.wx, grid_dx.wy, 32,
                                                B, C, OC);
 


                                                

  // dim3 block_size(TILE_SIZE, TILE_SIZE);
  // dim3 grid_size(std::ceil((float)OC/(float)TILE_SIZE), std::ceil((float)B/(float)TILE_SIZE));
  // embeddingln_backward_kernel<<<grid_size, block_size, 0, main_stream>>>(x, dW, dy, TILE_SIZE, B, C, OC);
}