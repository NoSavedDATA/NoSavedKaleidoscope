

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../elementwise_kernels_inline.cu"


__global__ void random_padding_cropping_kernel(
  const float* input,
  float* output,
  int batch_size,
  int channels,
  int input_height,
  int input_width,
  int crop_height,
  int crop_width,
  int padding,
  unsigned long long seed)
{
  int b = blockIdx.x;
  int c = blockIdx.y;
  int hw = blockIdx.z * blockDim.x + threadIdx.x;

  int y = hw / input_width;
  int x = hw % input_width;


  //int y = threadIdx.y;
  //int x = threadIdx.x;

  if (b >= batch_size || c >= channels || y >= crop_height || x >= crop_width)
    return;

  // Setup random number generator
  curandState state;
  curand_init(seed, b, 0, &state); // one random state per batch

  // Generate random padding values
  int pad_top = curand(&state) % (padding + 1); // Range belongs to [0, padding]
  int pad_bottom = curand(&state) % (padding + 1);
  int pad_left = curand(&state) % (padding + 1);
  int pad_right = curand(&state) % (padding + 1);

  // Compute the padded height and width
  int padded_height = input_height + pad_top + pad_bottom;
  int padded_width = input_width + pad_left + pad_right;

  // Ensure crop dimensions fit within padded dimensions
  int max_crop_top = padded_height - crop_height;
  int max_crop_left = padded_width - crop_width;

  // Generate random crop starting points
  int crop_top = curand(&state) % (max_crop_top + 1);
  int crop_left = curand(&state) % (max_crop_left + 1);

  // Compute input coordinates for any location within the crop
  int input_y = crop_top + y;
  int input_x = crop_left + x;

  // Handle padding and out-of-bounds conditions
  if (input_y >= pad_top && input_y < padded_height - pad_bottom &&
      input_x >= pad_left && input_x < padded_width - pad_right) {
    // Within padded region, read from input
    output[b * channels * crop_height * crop_width + c * crop_height * crop_width + y * crop_width + x] =
      input[b * channels * input_height * input_width + c * input_height * input_width + (input_y - pad_top) * input_width + (input_x - pad_left)];
  } else {
    // Outside padded region, set to zero
    output[b * channels * crop_height * crop_width + c * crop_height * crop_width + y * crop_width + x] = 0.0f;
  }
}


__global__ void random_horizontal_flip_kernel(const float *input_tensor, float *output_tensor, 
                                              int batch_size, int channels, int height, int width, 
                                              unsigned long long seed) {
  // Thread ID
  int batch_idx = blockIdx.x;
  int channel_idx = blockIdx.y;
  int hw = blockIdx.z * blockDim.x + threadIdx.x;

  int h = hw / width;
  int w = hw % width;

  int idx = ((batch_idx * channels + channel_idx) * height + h) * width + w;

  if (idx > (batch_size*channels*height*width))
    return;

  // Random number generator initialization
  curandState state;
  curand_init(seed, batch_idx, 0, &state);
    

  // Determine if the current image should be flipped
  bool flip = curand_uniform(&state) > 0.5f;

  if (flip) {
    // Compute the flipped column index
    int flipped_col_idx = width - w - 1;
    int flipped_idx = ((batch_idx * channels + channel_idx) * height + h) * width + flipped_col_idx;
    output_tensor[idx] = input_tensor[flipped_idx];
  } else {
    output_tensor[idx] = input_tensor[idx];
  }
}


__global__ void normalize_img_kernel(float *output_tensor, const float *input_tensor, 
                                     const float *mean, const float *std,
                                     int batch_size, int channels, int height, int width) {
  // Thread ID
  int b = blockIdx.x;
  int c = blockIdx.y;
  int hw = blockIdx.z * blockDim.x + threadIdx.x;
  
  int h = hw / width;
  int w = hw % width;

  //int idx = ((batch_idx * channels + c) * height + h) * width + w;
  int idx = b * channels * height * width + c * height * width + h * width + w;
    
  output_tensor[idx] = (input_tensor[idx]-mean[c])/std[c];    
}


__global__ void jitter_kernel(float *y, const float *x, const float factor, const int dims_prod,
                               unsigned long long seed)
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx>dims_prod)
    return;


  curandState state;
  curand_init(seed, idx, 0, &state);

  float r = curand_normal(&state);
  r = cuda_clip(r, -2.0, 2.0);

  y[idx] = x[idx] * (1+factor*r);
}
