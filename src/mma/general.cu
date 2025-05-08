#include <cuda_runtime.h>

#include "../common/cu_commons.h"
#include "../cuda_threads/include.h"
#include "../cuda_kernels/handles.h"
#include "../tensor/include.h"
#include "utils.h"

using namespace nvcuda;



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






void matmul_backward2(
  // float *inp,  float *weight,
  // int B, int C, int OC,
  DT_tensor *L_tensor, DT_tensor *R_tensor,
  float *dinp, float *dw,
  float *dout)
{

  float *inp = L_tensor->tensor_ptr;
  float *weight = R_tensor->tensor_ptr;

  std::vector<float> BC = format_LinearLayer_Dims(L_tensor->dims);
  float B  = BC[0];
  float C  = BC[1];
  float OC = R_tensor->dims[0]; 


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

  cudaStreamSynchronize(main_stream->stream);

  mult_backwarddx<<<grid_size, block_size, shared_mem_size>>>(weight, dinp, dout, TILE_SIZE, TILE_SIZE_SQ, B, C, OC);

  RegisterEvent(dx_stream);


  dim3 grid_size2(std::ceil(C/(float)TILE_SIZE), std::ceil(OC/(float)TILE_SIZE));
  mult_backwarddw_acc<<<grid_size2, block_size, shared_mem_size>>>(inp, dw, dout, TILE_SIZE, TILE_SIZE_SQ, B, C, OC);



  //PrintTensorF(dw, OC, C);

  StreamAwaitStreamB(main_stream->stream, dx_stream);
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
            
  gemm_operator_dx(main_stream->stream);
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
            
  gemm_operator_dw(main_stream->stream);
  gemm_operator_dw(args_dw);
  */
}

void matmul_forward(float* out,
                     float* inp, float* W,
                     int B, int C, int OC, int thread_id) {
        
  const float alpha = 1.0f;
  const float beta = 0.0f;
  

  //std::cout << "matmul forward. B: " << B << " C: " << C << " OC: " << OC << "\n";


  if (thread_id==0)
  {
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B, C, &alpha, W, C, inp, C, &beta, out, OC));
    
    cudaStream_t stream = ThreadsStream[thread_id];


    

    // dim3 block_size(TILE_SIZE, TILE_SIZE);
    // dim3 grid_size(std::ceil(OC/(float)TILE_SIZE), std::ceil(B/(float)TILE_SIZE));
    // int shared_mem_size = 2*TILE_SIZE*TILE_SIZE*sizeof(float);
    // mult_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(inp, W, out, TILE_SIZE, TILE_SIZE*TILE_SIZE, B, C, OC);
   

    // constexpr int num_warps_x{4};
    // constexpr int num_warps_y{4};
    

    // constexpr int WMMA_T{16};
    // dim3 block_size(num_warps_x * WARP_SIZE, num_warps_y);
    // dim3 grid_size(std::ceil((OC + (num_warps_x*WMMA_T - 1)) / (float)(num_warps_x*WMMA_T)), std::ceil((B + (num_warps_y*WMMA_T - 1)) / (float)(num_warps_y*WMMA_T)));

    // int shared_mem_size = num_warps_y*WMMA_T*WMMA_T*num_warps_x*sizeof(float);


    // // float *bank;
    // // cudaMalloc(&bank, 32*16*sizeof(float));
    
    // // set_to_one_kernel<<<16, 32,0,stream>>>(bank, 16*32);
    
    // wmma_mult_kernel<WMMA_T,num_warps_x,num_warps_y><<<grid_size, block_size, shared_mem_size, stream>>>(inp, W, out, B, C, OC);


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
                              
  gemm_operator(main_stream->stream);
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