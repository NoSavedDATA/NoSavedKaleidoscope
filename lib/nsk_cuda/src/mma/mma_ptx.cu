

#include "utils.h"

using namespace nvcuda;

#define MMA_PTX_M 16
#define MMA_PTX_N 8
#define MMA_PTX_K 16



template<int WMMA_T, int wk>
__global__ void mma_ptx(const float *__restrict__ x, const float *__restrict__ w,
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

    int warp_y = warpId / bx_per_wx;
    int warp_x = warpId % bx_per_wx;

    int mw = laneId / 4;
    int ml = laneId % 4;





  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> x_frag[2];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> w_frag[2];

    


    
    const int wy_count = 2;
    const int wy_loop = 2;


    // __half x_frag[4*4];
    // __half w_frag[2*2];
    float acc_frag[4*4*8];
    

    for (int i=0; i<4*4*8; ++i)
    {
        acc_frag[i] = 0.0f;
    }





    extern __shared__ float smem[];
    float *out_smem = smem;
    

    float *x_smem     = smem;
    float *w_smem     = smem + by*wk;
    
    
    // index auxiliars
    int xor_addr = smem_xor_cp_async(laneId);

    int by_warp_offset = by_per_w*warpId;
    int bx_warp_offset = bx_per_w*warpId;


    #pragma unroll
    for (int tile=0; tile<C; tile+=wk)
    {
        // warp * mw_size * i_size + mw*i_size + i
        
        
        

        



        // Each iter processes 4/ml rows and 8/mw*(4 floats) | 32 cols.
        // So, we jump 4|ml rows per iter.

        for(int block_tile=0; block_tile<by_per_w/4; ++block_tile)
        {
            // int row_aux = warp_x*4 + ml;
            // int row = block_y*by + y_blocking*wy + warp_y*WMMA_T + row_aux;

            int row = block_y*by + by_warp_offset + block_tile*4 + ml;
            float const *gmem_ptr = x + row*C + tile+mw*4;

            gmem_to_smem_xor(gmem_ptr,  *(x_smem + (by_warp_offset + block_tile*4)*wk + xor_addr),
                             (row<B) ? std::min((( C-(tile+mw*4)) /4)*4, 16) : 0);            
        }




        for(int block_tile=0; block_tile<bx_per_w/4; ++block_tile)
        {

            int row = block_x*bx + bx_warp_offset + block_tile*4 + ml;
            float const *gmem_ptr = w + row*C + tile+mw*4;

            gmem_to_smem_xor(gmem_ptr,  *(w_smem + (bx_warp_offset + block_tile*4)*wk + xor_addr),
                             (row<OC) ? std::min((( C-(tile+mw*4)) /4)*4, 16) : 0);            
        }







        asm volatile("cp.async.wait_all;");
        __syncthreads();





        

        for (int k_stride=0; k_stride<2; ++k_stride)
        {



            for (int wy_tile=0; wy_tile<wy_per_wmma_n; ++wy_tile)
            {
                
                const float *smem_off = x_smem + (warp_y*wy + wy_tile*MMA_PTX_M);

                // int wi = i/4;
                // int xi = i%4;

                // int xj = j/4;
                // int wj = j%4;

                // int offset = smem_dexor_from_cp_async(xi, xj*2 + k_stride)+wj;


                // ld_smem_to_reg_A(x_frag, smem_off);
            }






            for (int wx_tile=0; wx_tile<wx_per_wmma_m; ++wx_tile)
            {



                for (int wy_tile=0; wy_tile<wy_per_wmma_n; ++wy_tile)
                {






                    // if ((block_y*by + y_blocking*wy + warp_y*WMMA_T)<B && (block_x*bx + x_blocking*wx + warp_x*WMMA_T)<OC)
                    // {
                            
                            // smem_xor_to_reg_A(x_frag, x_smem + (y_blocking*wy + warp_y*WMMA_T)*wk, wk, k_stride);
                            // if (y_blocking==0)
                            //     smem_xor_to_reg_B(w_frag, w_smem + (x_blocking*wx + warp_x*WMMA_T)*wk, wk, k_stride);
                            
                            
                            // wmma16x16x16(acc_frag+(x_blocking*by_per_w+y_blocking)*8, x_frag, w_frag);
                        

                    // }
                }
            }
        }

    }
    

    





    // float *_out = out_smem + warp_y*WMMA_T*(X_WARPS*WMMA_T) + warp_x*WMMA_T;

    

    // for (int x_blocking=0; x_blocking<bx_per_w; ++x_blocking)
    // {
    //     for (int y_blocking=0; y_blocking<by_per_w; ++y_blocking)
    //     {
    //         __syncthreads();

    //         int threaded_row = block_y*by + y_blocking*wy + warp_y*WMMA_T;
    //         int threaded_col = block_x*bx + x_blocking*wx + warp_x*WMMA_T;

    //         if (threaded_row<B && threaded_col<OC && (warp_y*WMMA_T)<B && (warp_x*WMMA_T)<OC)
    //         {
                
                
    //             frag_to_mem(acc_frag+(x_blocking*by_per_w+y_blocking)*8, _out, X_WARPS*WMMA_T);
                
                
                

    //         #pragma unroll
    //             for (int tile=0; tile<std::ceil((WMMA_T*WMMA_T)/(float)(warpSize)); ++tile)
    //             {
    //             int tile_idx = tile*warpSize + laneId;

    //             int row =  tile_idx / WMMA_T;
    //             int col = (tile_idx % WMMA_T);


    //             if((threaded_row+row)<B  &&  (threaded_col+col)<OC && row<WMMA_T)
    //                 out[(threaded_row+row)*OC + threaded_col+col] = _out[row*(X_WARPS*WMMA_T)+col];

    //             }
    //         }
    //     }
    // }
}








template<int WMMA_T>
inline void blocking_mma_ptx(const float *x, const float *w, float *o, int B, int C, int OC, cudaStream_t stream)
{
  Wmma_Grid grid = CalculateBlockingSize(OC, B,
                                         8,
                                         128, 64,
                                         32, 32,
                                         16, 16);
    // std::cout << "OC: " << OC << ", B: " << B << \
    //   "\ngx: " << grid.g.x << ", gy: " << grid.g.y << ", bx: " << grid.b.x << ", by: " << grid.b.y << \
    //   "\nblocking warps per block x: " << grid.bx_per_w << ", y: " << grid.by_per_w << \
    //   "\nx warps: " << grid.w.x/32 << ", y warps: " << grid.w.y <<  "\n\n";

    mma_ptx<WMMA_T, 32><<<grid.g, grid.w, grid.smem, stream>>>
                            (x, w, o, B, C, OC, grid.bx, grid.by, grid.wx, grid.wy,
                            grid.bx_per_w, grid.by_per_w,
                            grid.bx_per_wx, grid.by_per_wy,
                            grid.wx_per_wmma_m, grid.wy_per_wmma_n);
}
