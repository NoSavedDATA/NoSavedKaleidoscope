#include <iostream>
#include <string>

#include <cuda_runtime.h>


#include "../../common/include.h"
#include "../../codegen/string.h"
#include "stb_lib.h"
#include "store.h"



extern "C" float save_img(int thread_id, Tensor *tensor, char *img_name)
{
  
  int c, h, w;

  c = tensor->dims[tensor->dims.size()-3];
  h = tensor->dims[tensor->dims.size()-2];
  w = tensor->dims[tensor->dims.size()-1];


  // Check if the tensor has 3 channels
  if (c != 3) {
    std::cerr << "Only 3-channel images are supported." << std::endl;
    return -1;
  }

  float *tensor_cpu = new float[c*h*w];
  cudaCheck(cudaMemcpy(tensor_cpu, tensor->tensor_ptr, c*h*w*sizeof(float), cudaMemcpyDeviceToHost));

  // Convert tensor to 8-bit format
  std::vector<uint8_t> imageData(h * w * c);
  for (int ch = 0; ch < c; ++ch) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        int tensorIndex = (ch * h + y) * w + x;
        int imageIndex = (y * w + x) * c + ch;
        imageData[imageIndex] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, tensor_cpu[tensorIndex] * 255.0f)));
      }
    }
  }

  std::string img = "/home/nosaveddata/imgs/";
  img = img + img_name;
  img = img + RandomString(4);
  img = img + ".png";

  std::cout << "writing image: " << img << "\n";

  // Write the image using stb_image_write
  if (!stbi_write_png(img.c_str(), w, h, c, imageData.data(), w * c)) {
    std::cerr << "Failed to write image to " << img_name << std::endl;
  }

  delete[] tensor_cpu;

  return 0;
}
