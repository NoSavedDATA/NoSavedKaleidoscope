#include "../nsk_cuda/include.h"
#include "utils.h"


using namespace nvcuda;

template<int WMMA_T, int wk>
__global__ void wmma_cp_async_blocking(const float *__restrict__ x, const float *__restrict__ w,
                      float *__restrict__ out, const int B, const int C, const int OC,
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


  
  const int wy_count = 2;
  const int wy_loop = 2;

  float   acc_frag[4*4*8];
  // float delta_frag[4*4*8];

  for (int i=0; i<4*4*8; ++i)
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



    asm volatile("cp.async.commit_group;\n" ::);

    // asm volatile("cp.async.wait_all;");
    // __syncthreads();




    for (int x_blocking=0; x_blocking<wx_per_bx; ++x_blocking)
    {
      if (x_blocking>0)
      {
        asm volatile("cp.async.wait_all;");
        __syncthreads();
      }

      for (int y_blocking=0; y_blocking<wy_per_by; ++y_blocking)
      {


        if (x_blocking==0 && (y_blocking+1)<wy_per_by)
        {

          if (row_aux1<WMMA_T)
          {
            int row = block_y*by + (y_blocking+1)*wy + warp_y*WMMA_T + row_aux1;
            float const *gmem_ptr = x + row*C + tile+mw*4;

            // extra *2 to accomodate 32 instead of 16 C (i.e, the whole warpSize)
            //       *4 is necessary as it needs to accomodate 4 consecutive floats
            gmem_to_smem_xor(gmem_ptr,  *(x_smem + ((y_blocking+1)*wy + warp_y*WMMA_T+ warp_x*4)*wk + xor_addr),  (row<B) ? std::min((( C-(tile+mw*4)) /4)*4, 16) : 0);
          }
          asm volatile("cp.async.commit_group;\n" ::);
        }


        if ((x_blocking+1)<wx_per_bx && (y_blocking+1)==wy_per_by)
        {
          for (int aux_i=0; aux_i<wy_loop; ++aux_i)
          {
            int row_aux2 = (aux_i*wy_count+warp_y)*4 + ml;

            if (row_aux2<WMMA_T)
            {
              int row = block_x*bx + (x_blocking+1)*wx + warp_x*WMMA_T + row_aux2;
              float const *gmem_ptr = w + row*C + tile+mw*4;
              
              gmem_to_smem_xor(gmem_ptr,  *(w_smem + ((x_blocking+1)*wx + warp_x*WMMA_T + (aux_i*wy_count + warp_y)*4)*wk + xor_addr),  (row<OC) ? std::min((( C-(tile+mw*4)) /4)*4, 16) : 0);        
            }
          }
          asm volatile("cp.async.commit_group;\n" ::);
        }



        if (x_blocking==0)
        {
          asm volatile("cp.async.wait_group %0;" ::"n"(1));
          __syncthreads();
        }



        if ((block_y*by + y_blocking*wy + warp_y*WMMA_T)<B && (block_x*bx + x_blocking*wx + warp_x*WMMA_T)<OC)
        {
          __syncwarp();
          for (int k_stride=0; k_stride<2; ++k_stride)
          {
            
            smem_xor_to_reg_A(x_frag, x_smem + (y_blocking*wy + warp_y*WMMA_T)*wk, wk, k_stride);
            if (y_blocking==0)
              smem_xor_to_reg_B(w_frag, w_smem + (x_blocking*wx + warp_x*WMMA_T)*wk, wk, k_stride);
            // smem_xor_to_reg_A_ec(x_frag, x_frag_delta, x_smem + (y_blocking*wy + warp_y*WMMA_T)*wk, wk, k_stride);
            // if (y_blocking==0)
            //   smem_xor_to_reg_B_ec(w_frag, w_frag_delta, w_smem + (x_blocking*wx + warp_x*WMMA_T)*wk, wk, k_stride);
            
            wmma16x16x16(acc_frag+(x_blocking*wy_per_by+y_blocking)*8, x_frag, w_frag);
            // wmma16x16x16(delta_frag+(x_blocking*wy_per_by+y_blocking)*8, x_frag_delta, w_frag);
            // wmma16x16x16(delta_frag+(x_blocking*wy_per_by+y_blocking)*8, x_frag, w_frag_delta);

          }
        }
      }
    }

  }
  

  



  float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;

  

  for (int x_blocking=0; x_blocking<wx_per_bx; ++x_blocking)
  {
    for (int y_blocking=0; y_blocking<wy_per_by; ++y_blocking)
    {
      __syncthreads();

      int threaded_row = block_y*by + y_blocking*wy + warp_y*WMMA_T;
      int threaded_col = block_x*bx + x_blocking*wx + warp_x*WMMA_T;

      if (threaded_row<B && threaded_col<OC && (warp_y*WMMA_T)<B && (warp_x*WMMA_T)<OC)
      {
        
        
        frag_to_mem(acc_frag+(x_blocking*wy_per_by+y_blocking)*8, _out, X_WARPS*WMMA_T);
        // frag_to_mem_ec(acc_frag+(x_blocking*wy_per_by+y_blocking)*8, delta_frag+(x_blocking*wy_per_by+y_blocking)*8, _out, X_WARPS*WMMA_T);
        
        

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
  }
}



template<int WMMA_T>
inline void wmma(const float *x, const float *w, float *o, int B, int C, int OC, cudaStream_t stream)
{
  // Grid grid = CalculateBlockingSize(OC, B);

  // wmma_cp_async_blocking<WMMA_T, 32><<<grid.g, grid.w, grid.smem, stream>>>
  //                         (x, w, o, B, C, OC, grid.b.x, grid.b.y, (grid.w.x/32)*WMMA_T, grid.w.y*WMMA_T,
  //                         grid.wx_per_bx, grid.wy_per_by, grid.w.x/32, grid.w.y);
}
