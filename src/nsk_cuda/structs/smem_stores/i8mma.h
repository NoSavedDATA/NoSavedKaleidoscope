#pragma once



template<int warp_rows_per_m, int warp_cols_per_n, typename T>
__device__ void smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n, T>::store_C_mma(float *out, float *out_smem, const float *scale_M, const float *scale_N, int threaded_row, int threaded_col,
                          int M, int N,
                          int WMMA_M, int WMMA_N, int WMMA_K) {
  #pragma unroll
    for (int tile=0; tile<std::ceil((WMMA_N*WMMA_M)/(float)(warpSize)); ++tile)
    {
      int tile_idx = tile*warpSize + wmma_idx.laneId;

      int row =  tile_idx / WMMA_M;
      int col = (tile_idx % WMMA_M);


      if((threaded_row+row)<M  &&  (threaded_col+col)<N && row<WMMA_K)
      {
        // if (blockIdx.x==0&&threadIdx.x==0)
        // {
        //   printf("Storing %f - %f\n", out_smem[row*(wmma_idx.bx_per_wx*WMMA_M)+col], out_smem[row*(wmma_idx.bx_per_wx*WMMA_M)+col] / (scale_M[threaded_row+row] * scale_N[threaded_col+col]));
        //   printf("Scale is: %f - %f\n", scale_M[threaded_row+row], scale_N[threaded_col+col]);
        // }
        
        out[(threaded_row+row)*N + threaded_col+col] = out_smem[row*(wmma_idx.bx_per_wx*WMMA_M)+col] / (scale_M[threaded_row+row] * scale_N[threaded_col+col]);
      }
    }
}


template<int warp_rows_per_m, int warp_cols_per_n, typename T>
__device__ void smem_cpasync_wmma_loader<warp_rows_per_m, warp_cols_per_n, T>::blocking_tiled_store_C_mma(float *out_tensor, const float *scale_M, const float *scale_N, int *out,
                                         int M, int N, const int WMMA_M, const int WMMA_N, const int WMMA_K)
{

    float *out_smem = smem + wmma_idx.warp_y*WMMA_M*(4*WMMA_N) + wmma_idx.warp_x*WMMA_N; // 4 from 4 col_warps
    // float *out_smem = smem + wmma_idx.warpId*WMMA_K;


    for (int wx_tile=0; wx_tile<warp_rows_per_m; ++wx_tile)
    {
      for (int wy_tile=0; wy_tile<warp_cols_per_n; ++wy_tile)
      {
        __syncthreads();


        int threaded_row = wmma_idx.block_y*wmma_idx.blocking_size_y + wmma_idx.warp_y*wmma_idx.wy + wy_tile*WMMA_M;
        int threaded_col = wmma_idx.block_x*wmma_idx.blocking_size_x + wmma_idx.warp_x*wmma_idx.wx + wx_tile*WMMA_N;

        if (threaded_row<M && threaded_col<N && (wmma_idx.warp_y*wmma_idx.wy)<M && (wmma_idx.warp_x*wmma_idx.wx)<N)
        {
          
        //   frag_to_mem(frag_loader.acc_frag+(wx_tile*warp_cols_per_n + wy_tile)*8, out_smem, 64);
          frag_to_mem(out+(wx_tile*warp_cols_per_n + wy_tile)*8, out_smem, 64);
          
          // if (threadIdx.x==0)
          //   printf("tile %d/%d Out frag has: %d, %d, %d, %d, %d, %d, %d, %d\n", wx_tile, wy_tile, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
          
          store_C(out_tensor, out_smem, scale_M, scale_N, threaded_row, threaded_col, M, N, WMMA_M, WMMA_N, WMMA_K);
        }
      }
    }
}