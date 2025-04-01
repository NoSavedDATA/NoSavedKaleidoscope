#pragma once


__global__ void mse_kernel(float *dy, const float* y_hat, const float* y,
                            const float scale, const float dims_prod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < dims_prod) {
        dy[idx] = 2 * (y_hat[idx] - y[idx]) * scale;
    }
}


__global__ void online_mse(float *out, const float *y_hat, const float *y_true, int N, int C) {
  
    const int warpsPerBlock = blockDim.x / warpSize;
    int tid = threadIdx.x;

    

    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    // one warp one row
    int row = blockIdx.x * warpsPerBlock + warpId;
    
    if (laneId >= C)
        return;

    if (row >= N)
        return;

    
    const float *x = y_hat  + row * C;
    const float *y = y_true + row * C;
    

    // merge calculating maxval and sumval in one loop
    // which is an arithmetic improvment from online softmax over normal softmax
    float sumval = 0.0f;

#pragma unroll
    for (int i = laneId; i < C; i += warpSize) {
        // when updating the maxval, dynamically updates the previous sumval by
        // multiplying e^{previous_maxval - current_maxval}

        sumval += powf(x[i]-y[i], 2);
    }

    // use warp functions instead of cooperative groups for better readibility
    // calculate the warp wised maxval and sumval
    float offsetSumval;

#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        __syncwarp();
        offsetSumval = __shfl_down_sync(0xFFFFFFFF, sumval, offset);
        
        sumval += offsetSumval;
    }


    sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);


    out[row] = sumval/C;
    
}


__global__ void mse_with_priorities_kernel(float *dy, const float* y_hat, const float* y, const float *is_w,
                            const float scale, const float dims_prod) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dims_prod) {
        dy[tid] = 2 * (y_hat[tid] - y[tid]) * scale * is_w[tid];
    }
}