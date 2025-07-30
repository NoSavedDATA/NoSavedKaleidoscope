#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>


using namespace nvcuda;


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