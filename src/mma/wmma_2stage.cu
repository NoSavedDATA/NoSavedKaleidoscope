#include "utils.h"

using namespace nvcuda;


#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16





template<int WMMA_T, int wk>
__global__ void wmma_blocking(const float *__restrict__ x, const float *__restrict__ w,
                      float *__restrict__ out, const int B, const int C, const int OC,
                      const int bx, const int by,
                      const int wx, const int wy,
                      const int bx_per_w,     const int by_per_w,
                      const int bx_per_wx,    const int by_per_wy,
                      const int wx_per_wmma_m, const int wy_per_wmma_n) {





  int block_x = blockIdx.x;
  int block_y = blockIdx.y;


  int warpId = threadIdx.x / warpSize;
  int laneId = (threadIdx.x) % warpSize;

  // bx = 256, wx = 64
  //bx_per_wx = 4
  int warp_y = warpId / bx_per_wx;
  int warp_x = warpId % bx_per_wx;

  int mw = laneId / 4;
  int ml = laneId % 4;


  int smem_cache_off = (bx + by)*wk;
  int smem_store_off = 0;
  int smem_load_off = smem_cache_off;


  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag[4]; // wy_per_wmma_n
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag[8]; // wx_per_wmma_m
  



  float   acc_frag[4*8*8];
  

  for (int i=0; i<4*8*8; ++i)
  {
    acc_frag[i] = 0.0f;
  }


  


  extern __shared__ float smem[];
  float *out_smem = smem;
  // __half *hsmem = reinterpret_cast<__half*>(smem + Y_WARPS*WMMA_T*(X_WARPS*WMMA_T));

  float *x_smem     = smem;
  float *w_smem     = smem + by*wk;
  
  
  // index auxiliars
  int xor_addr = smem_xor_cp_async(laneId);

  int by_warp_offset = by_per_w*warpId;
  int bx_warp_offset = bx_per_w*warpId;





/////////////////////////////




  for(int block_tile=0; block_tile<by_per_w/4; ++block_tile) // 4 from 4 jumped rows
  {

      int row = block_y*by + by_warp_offset + block_tile*4 + ml; // 4 from 4 loaded bits
      float const *gmem_ptr = x + row*C + mw*4; // 4 from 4 loaded bits

      gmem_to_smem_xor(gmem_ptr,  *(x_smem + (by_warp_offset + block_tile*4)*wk + xor_addr), // 4 from 4 loaded bits
                        (row<B) ? std::min((( C-(mw*4)) /4)*4, 16) : 0);            
  }




  for(int block_tile=0; block_tile<bx_per_w/4; ++block_tile)
  {

      int row = block_x*bx + bx_warp_offset + block_tile*4 + ml;
      float const *gmem_ptr = w + row*C + mw*4;

      gmem_to_smem_xor(gmem_ptr,  *(w_smem + (bx_warp_offset + block_tile*4)*wk + xor_addr),
                        (row<OC) ? std::min((( C-(mw*4)) /4)*4, 16) : 0);            
  }




  asm volatile("cp.async.commit_group;\n" ::);

  // asm volatile("cp.async.wait_all;");
  // __syncthreads();




#pragma unroll
  for (int tile=0; tile<C; tile+=wk)
  {
    // warp * mw_size * i_size + mw*i_size + i
    


    smem_store_off ^= smem_cache_off;
    smem_load_off  ^= smem_cache_off;



    // if ((block_x+block_y+laneId+warp_x)==0)
    //   printf("load %d, \t\t store: %d\n", smem_load_off, smem_store_off);


    // Each iter processes 4/ml rows and 8/mw*(4 floats) | 32 cols.
    // So, we jump 4|ml rows per iter.


    int next_tile = tile + wk;

    if (next_tile<C)
    {
      
      for(int block_tile=0; block_tile<by_per_w/4; ++block_tile)
      {

          int row = block_y*by + by_warp_offset + block_tile*4 + ml;
          float const *gmem_ptr = x + row*C + next_tile+mw*4;

          gmem_to_smem_xor(gmem_ptr,  *(x_smem + smem_store_off + (by_warp_offset + block_tile*4)*wk + xor_addr),
                            (row<B) ? std::min((( C-(next_tile+mw*4)) /4)*4, 16) : 0);
      }




      for(int block_tile=0; block_tile<bx_per_w/4; ++block_tile)
      {

          int row = block_x*bx + bx_warp_offset + block_tile*4 + ml;
          float const *gmem_ptr = w + row*C + next_tile+mw*4;

          gmem_to_smem_xor(gmem_ptr,  *(w_smem + smem_store_off + (bx_warp_offset + block_tile*4)*wk + xor_addr),
                            (row<OC) ? std::min((( C-(next_tile+mw*4)) /4)*4, 16) : 0);
      }

  
      asm volatile("cp.async.commit_group;\n" ::);
      asm volatile("cp.async.wait_group %0;" ::"n"(1));
    } else {
      asm volatile("cp.async.wait_all;");
    }
    

    __syncthreads();







/////////////////////////////


    for (int k_stride=0; k_stride<2; ++k_stride)
    {


      for (int wy_tile=0; wy_tile<wy_per_wmma_n; ++wy_tile)
      {

        // if ((block_x+block_y+laneId+warp_x)==0)
        //   printf("wy tile %d, wy_per_wmma_n: %d, warp y: %d, row: %d\n", wy_tile, wy_per_wmma_n, warp_y, (warp_y*wy + wy_tile*WMMA_N));
        // __syncthreads();
        smem_xor_to_reg_A(x_frag[wy_tile], x_smem + smem_load_off + (warp_y*wy + wy_tile*WMMA_N)*wk, wk, k_stride);
      }


      for (int wx_tile=0; wx_tile<wx_per_wmma_m; ++wx_tile)
      {

        // if ((block_x+block_y+laneId+warp_x+warp_y)==0)
        //   printf("wx tile %d, wx_per_mma_n: %d\n", wx_tile, wx_per_wmma_m);
        // __syncthreads();
        smem_xor_to_reg_B(w_frag[wx_tile], w_smem + smem_load_off + (warp_x*wx + wx_tile*WMMA_M)*wk, wk, k_stride);
      }
      



/////////////////////////////



      int wy_tile=0;
      int jump=1;
      int gate=1;
      for (int wx_tile=0; wx_tile<wx_per_wmma_m; ++wx_tile)
      {
        
        for (wy_tile=0; wy_tile<wy_per_wmma_n; wy_tile+=1)
        {
          if ((block_y*by + wy_tile*WMMA_N)<B && (block_x*bx + wx_tile*WMMA_M)<OC)
            wmma16x16x16(acc_frag+(wx_tile*wy_per_wmma_n + wy_tile)*8, x_frag[wy_tile], w_frag[wx_tile]); // 8 is the frag ld

           
        }

        // for (; wy_tile*jump<(wy_per_wmma_n*gate); wy_tile+=jump)
        // {


        //   if ((block_y*by + wy_tile*WMMA_N)<B && (block_x*bx + wx_tile*WMMA_M)<OC)
        //   {

        //     wmma16x16x16(acc_frag+(wx_tile*wy_per_wmma_n + wy_tile)*8, x_frag[wy_tile], w_frag[wx_tile]);
            
        //   }
        // }

        // jump*=-1;
        // gate = (jump+1)/2;
      }
      __syncthreads();
    }

    // asm volatile("cp.async.wait_all;");
    // __syncthreads();

  }
  



/////////////////////////////



  // float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;
  float *_out = out_smem + warp_y*WMMA_N*(bx_per_wx*WMMA_T) + warp_x*WMMA_M; // todo: is this correct?
  


  for (int wx_tile=0; wx_tile<wx_per_wmma_m; ++wx_tile)
  {

    for (int wy_tile=0; wy_tile<wy_per_wmma_n; ++wy_tile)
    {
      __syncthreads();


      int threaded_row = block_y*by + warp_y*wy + wy_tile*WMMA_N;
      int threaded_col = block_x*bx + warp_x*wx + wx_tile*WMMA_M;

      if (threaded_row<B && threaded_col<OC && (warp_y*wy)<B && (warp_x*wx)<OC)
      {
        
        
        frag_to_mem(acc_frag+(wx_tile*wy_per_wmma_n + wy_tile)*8, _out, bx_per_wx*WMMA_T);
        
        
        

    #pragma unroll
        for (int tile=0; tile<std::ceil((WMMA_N*WMMA_M)/(float)(warpSize)); ++tile)
        {
          int tile_idx = tile*warpSize + laneId;

          int row =  tile_idx / WMMA_M;
          int col = (tile_idx % WMMA_M);


          if((threaded_row+row)<B  &&  (threaded_col+col)<OC && row<WMMA_T)
            out[(threaded_row+row)*OC + threaded_col+col] = _out[row*(bx_per_wx*WMMA_M)+col];

        }
      }
    }
  }
}












template<int WMMA_T>
inline void blocking_mma_2stage(const float *x, const float *w, float *o, int B, int C, int OC, cudaStream_t stream)
{
  Grid2 grid = CalculateBlockingSize2(OC, B);

  // std::cout << "OC: " << OC << ", B: " << B << \
  //   "\ngx: " << grid.g.x << ", gy: " << grid.g.y << ", bx: " << grid.b.x << ", by: " << grid.b.y << \
  //   "\nblocking warps per block x: " << grid.bx_per_w << ", y: " << grid.by_per_w << \
  //   "\nx warps: " << grid.w.x/32 << ", y warps: " << grid.w.y <<  "\n\n";

  wmma_blocking<WMMA_T, 32><<<grid.g, grid.w, grid.smem, stream>>>
                          (x, w, o, B, C, OC, grid.b.x, grid.b.y, grid.wx, grid.wy,
                          grid.bx_per_w, grid.by_per_w,
                          grid.bx_per_wx, grid.by_per_wy,
                          grid.wx_per_wmma_m, grid.wy_per_wmma_n);
}
