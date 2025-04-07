#pragma once




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
  unsigned long long seed);


__global__ void random_horizontal_flip_kernel(const float *input_tensor, float *output_tensor, 
                                              int batch_size, int channels, int height, int width, 
                                              unsigned long long seed);

                                              
__global__ void normalize_img_kernel(float *output_tensor, const float *input_tensor, 
                                     const float *mean, const float *std,
                                     int batch_size, int channels, int height, int width); 


__global__ void jitter_kernel(float *y, const float *x, const float factor, const int dims_prod,
                               unsigned long long seed);
