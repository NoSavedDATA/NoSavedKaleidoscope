#include <mma.h>

using namespace nvcuda;

// __inline__ __device__ uint32_t cast_smem_ptr_to_uint(void *smem_ptr) {
//     return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
// }


__inline__ __device__ int smem_xor_cp_async(int lane_id) {
    // *4 means we calculate the storage for 16B/128b words (4 floats), done for each thread
    return ((lane_id % 8) ^ (lane_id / 8) + (lane_id / 8)*8) * 4;
}


__inline__ __device__ int smem_dexor_from_cp_async(int strided, int contiguous) {
    int stride=8;
    
    int tc = contiguous / 8;
    int ts = strided / 4;

    int c = contiguous % 8;
    int s = strided % 4;

    int k_index = c / 2;

    int bank = ((c & 1) * 4) | (s ^ k_index); // e [0, 7]
    
    int offset = tc * 32 + bank + (ts * 4 + k_index) * stride;


    // *4 means we calculate the storage for 16B/128b words (4 floats), done for each thread
    return offset*4;
}


__inline__ __device__ int smem_dexor_from_cp_async_i8(int strided, int contiguous) {
    int stride=8;
    
    int tc = contiguous / 8;
    int ts = strided / 4;

    int c = contiguous % 8;
    int s = strided % 4;

    int k_index = c / 2;

    int bank = ((c & 1) * 4) | (s ^ k_index); // e [0, 7]
    
    int offset = tc * 32 + bank + (ts * 4 + k_index) * stride;


    // *4 means we calculate the storage for 16B/128b words (4 floats), done for each thread
    return offset*4;
}

