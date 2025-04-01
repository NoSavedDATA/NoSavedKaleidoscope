
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>




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

