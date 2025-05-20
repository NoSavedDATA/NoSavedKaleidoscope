// #include <cuda_runtime.h>

// #include "../common/cu_commons.h"
// #include "../cuda_kernels/handles.h"
// #include "../tensor/include.h"
// #include "utils.h"

// using namespace nvcuda;



// __global__ void btc_mult_kernel(float *out, const float *x, const float *w, const int B, const int Tx, const int Tw, const int C, const int tile_size, const int tile_offset)
// {

//   int tx = threadIdx.x;
//   int ty = threadIdx.y;

//   int row = blockIdx.y * blockDim.y + threadIdx.y; // [B, T]
//   int col = blockIdx.x * blockDim.x + threadIdx.x; // [T]


//   //int b = row / Tx;
//   //int t = row % Tx;
//   int t = row;


//   extern __shared__ float smem_x[];
//   float *smem_w = smem_x + tile_offset;



//   for (int b=0; b<B; ++b)
//   {
//     float y=0;

//     for (int i=0; i < ceilf(C/(float)tile_size); ++i)
//     {
//       int _col1 = i*tile_size + tx;
//       int _col2 = i*tile_size + ty;

//       if (b<B && t<Tx && _col1<C)
//         smem_x[tx*tile_size + ty] = x[(b*Tx + t)*C + _col1];
//       else
//         smem_x[tx*tile_size + ty] = 0;


//       if (b<B && col<Tw && _col2<C)
//         smem_w[ty*tile_size + tx] = w[(b*Tx + col)*C + _col2];
//       else
//         smem_w[ty*tile_size + tx] = 0;

//       __syncthreads();

//       for(int j=0; j<tile_size; ++j)
//         y += smem_x[j* tile_size + ty] * smem_w[j* tile_size + tx];
      
//       __syncthreads();
//     }


//     // [B, T, T]
//     if(b<B && t<Tx && col<Tw)
//       out[(b*Tx + t)*Tw + col] = y;
//   }
// }



// extern "C" DT_tensor *btc_mult(int thread_id, DT_tensor *x, DT_tensor*w)
// {

//   std::vector<int> Ldims, Rdims, new_dims;
//   Ldims = x->dims;
//   Rdims = w->dims;
//   float *device_x = x->tensor_ptr;
//   float *device_w = w->tensor_ptr;

  

//   int B, Tx, C, Tw;

//   B  = Ldims[0];
//   Tx = Ldims[1];
//   C  = Ldims[2];

//   Tw = Rdims[1];


//   new_dims = {(float)B, (float)Tx, (float)Tw};
//   int new_dims_prod = DimsProd(new_dims);


//   float* device_y = get_from_pool(thread_id, new_dims_prod, "btc mult");

//   x->Sync();
//   w->Sync();



//   dim3 block_size(TILE_SIZE, TILE_SIZE);
//   dim3 grid_size(std::ceil(Tw/(float)TILE_SIZE), std::ceil((Tx)/(float)TILE_SIZE));
//   int shared_mem_size = 2*TILE_SIZE*TILE_SIZE*sizeof(float);

//   cudaStream_t stream = ThreadsStream[thread_id];


  
//   btc_mult_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(device_y, device_x, device_w, B, Tx, Tw, C, TILE_SIZE, TILE_SIZE_SQ);
//   //mult_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(device_x, device_w, device_y, TILE_SIZE, TILE_SIZE*TILE_SIZE, Tw, Tx, C);
  

//   DT_tensor *new_tensor = createTensor(device_y, new_dims, new_dims_prod, false, "");
//   new_tensor->AttrNodes(x, w, mult_op);
//   return new_tensor;
// }






// __global__ void btc_mult_kernelT(float *out, const float *x, const float *w, const int B, const int Tx, const int Tw, const int C, const int tile_size, const int tile_offset)
// {

//   int tx = threadIdx.x;
//   int ty = threadIdx.y;

//   int row = blockIdx.y * blockDim.y + threadIdx.y; // [Tx]
//   int col = blockIdx.x * blockDim.x + threadIdx.x; // [C]



//   //int b = row / Tx;
//   //int t = row % Tx;


//   extern __shared__ float smem_x[];
//   float *smem_w = smem_x + tile_offset;




  
//   // [B, Tx, Tw], consider T as Tw, row as Tx

//   for (int b=0; b<B; ++b)
//   {
//     float y=0;
//     __syncthreads();

//     for (int i=0; i < ceilf(Tw/(float)tile_size); ++i)
//     {
//       int _col1 = i*tile_size + tx;
//       int _col2 = i*tile_size + ty;

//       if (b<B && row<Tx && _col1<Tw)
//         smem_x[tx*tile_size + ty] = x[b*Tx*Tw + row*Tw + _col1];
//       else
//         smem_x[tx*tile_size + ty] = 0;


//       if (b<B && _col2<Tw && col<C)
//         smem_w[ty*tile_size + tx] = w[b*Tw*C + _col2*C + col];
//       else
//         smem_w[ty*tile_size + tx] = 0;

//       __syncthreads();

//       for(int j=0; j<tile_size; ++j)
//         y += smem_x[j*tile_size + ty] * smem_w[j*tile_size + tx];
      
//       __syncthreads();
//     }


//     // [B, Tx, C]
//     if (b<B && row<Tx && col<C)
//       out[b*Tx*C + row*C + col] = y;
      
//   }
// }



// extern "C" DT_tensor *btc_multT(int thread_id, DT_tensor *x, DT_tensor*w)
// {


//   std::vector<int> Ldims, Rdims, new_dims;
//   Ldims = x->dims;
//   Rdims = w->dims;
//   float *device_x = x->tensor_ptr;
//   float *device_w = w->tensor_ptr;

  

//   int B, Tx, Tw, C;

//   B  = Ldims[0];
//   Tx = Ldims[1];
//   Tw = Ldims[2];

//   C  = Rdims[2];


//   new_dims = {(float)B, (float)Tx, (float)C};
//   int new_dims_prod = DimsProd(new_dims);


//   float* device_y = get_from_pool(thread_id, new_dims_prod, "cuda mult");



//   dim3 block_size(TILE_SIZE, TILE_SIZE);
//   dim3 grid_size(std::ceil(C/(float)TILE_SIZE), std::ceil(Tx/(float)TILE_SIZE));
//   int shared_mem_size = 2*TILE_SIZE_SQ*sizeof(float);

//   cudaStream_t stream = ThreadsStream[thread_id];
  
//   btc_mult_kernelT<<<grid_size, block_size, shared_mem_size, stream>>>(device_y, device_x, device_w, B, Tx, Tw, C, TILE_SIZE, TILE_SIZE_SQ);


//   DT_tensor *new_tensor = createTensor(device_y, new_dims, new_dims_prod, false, "");
//   new_tensor->AttrNodes(x, w, mult_op);
//   return new_tensor;
// }