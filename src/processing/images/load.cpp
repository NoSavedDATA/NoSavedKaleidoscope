#include <iostream>

#include "../../common/cu_commons.h"
#include "../../compiler_frontend/logging.h"
#include "../../mangler/scope_struct.h"
#include "../../tensor/tensor_dim_functions.h"
#include "stb_lib.h"
#include "load.h"
#include "interpolate.h"


extern "C" float *load_img(Scope_Struct *scope_struct, char *img_name)
{
  int width, height, channels;
  
  unsigned char* image_data = stbi_load(img_name, &width, &height, &channels, 0);

  if (image_data) {
    

    float *image_data_float = new float[width * height * channels];
  
    // Loop through each pixel and convert to float between 0.0 and 1.0
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
          // Assuming unsigned char has 8 bits, scale by 1/255.0 to get a float value between 0.0 and 1.0
          //image_data_float[(y * width + x) * channels + c] = (float)image_data[(y * width + x) * channels + c] / 255.0f;

          // Convert BHWC into BCHW
          image_data_float[c * (height * width) + y * width + x] = (float)image_data[(y * width + x) * channels + c] / 255.0f;
        }
      }
    }

    stbi_image_free(image_data);
    return image_data_float;
    
  } else {
    std::string img_n = img_name;
    std::string _error = "Failed to open image: " + img_n + ".";
    LogErrorS(_error);
  }

  return nullptr;
}


extern "C" float * gload_img(Scope_Struct *scope_struct, DT_tensor tensor, char *img_name, int batch_idx)
{
  //std::cout << "LOADING IMAGE FOR: " << tensor.name <<  "\nImage: " << img_name << "\n";
  

  int width, height, channels;
  
  unsigned char* image_data = stbi_load(img_name, &width, &height, &channels, 0);

  if (image_data) {
    
    std::vector<int> dims = tensor.dims;

    //std::cout << "GLOAD IMG, dims of " << tensor.name << "\n";
    //PrintDims(dims);

    std::vector<int> batchless_dims = BatchLessDims(dims);
    int batchless_dims_prod = DimsProd(batchless_dims);
    

    if (batchless_dims_prod < width*height*channels)
    {
      std::string t_n = tensor.name;
      std::string _error = "The image dimensions are incompatible with the tensor " + t_n + " dimensions.";
      LogErrorS(_error);


      std::cout << "\nTENSOR BATCHLESS DIMS:" << "\n";
      PrintDims(batchless_dims);

      std::cout << "\nImage required dims: [" << width << ", " << height << ", " << channels << "]\n\n";

      return nullptr;
    }
    if (batch_idx > dims[0])
    {
      std::string _error = "Tried to load a pinned tensor on batch index " + std::to_string(batch_idx) + ", whereas this tensor batch size is " + std::to_string(dims[0]) + ".";
      LogErrorS(_error);
    }



    //float *image_data_float = new float[width * height * channels];
    float *image_data_float = tensor.cpu_tensor_ptr;
    int batch_offset =  batchless_dims_prod*batch_idx;
    //std::cout << "batch idx: " << batch_idx << ", batch offset: " << batch_offset << "\n";
  
    // Loop through each pixel and convert to float between 0.0 and 1.0
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
          // Assuming unsigned char has 8 bits, scale by 1/255.0 to get a float value between 0.0 and 1.0
          //image_data_float[batch_offset + (y * width + x) * channels + c] = (float)image_data[(y * width + x) * channels + c] / 255.0f;
          image_data_float[batch_offset + c * (height * width) + y * width + x] = (float)image_data[(y * width + x) * channels + c] / 255.0f;
        }
      }
    }
    stbi_image_free(image_data);

    return image_data_float;
    
  } else {
    std::string img_n = img_name;
    std::string _error = "Failed to open image: " + img_n + ".\n\n";
    LogErrorS(_error);
  }

  return nullptr;
}







extern "C" float * wload_img(Scope_Struct *scope_struct, DT_tensor *tensor, char *img_name, int worker_idx, int batch_idx)
{
  //std::cout << "LOADING IMAGE FOR: " << tensor->name <<  "\n";
  //std::cout << "Image: " << img_name <<  "\n";


  int width, height, channels;

  //std::cout << "GLOAD IMG, dims of " << img_name << "\n";
  
  unsigned char* image_data = stbi_load(img_name, &width, &height, &channels, 0);

  if (image_data) {
    
    std::vector<int> dims = tensor->dims;

    
    
    std::vector<int> workerless_dims = BatchLessDims(dims);
    int workerless_dims_prod = DimsProd(workerless_dims);

    std::vector<int> batchless_dims = BatchLessDims(workerless_dims);
    int batchless_dims_prod = DimsProd(batchless_dims);
    


    if (batchless_dims_prod < width*height*channels)
    {
      std::string t_n = tensor->name;
      std::string _error = "The image dimensions are incompatible with the tensor \033[95m" + t_n + "\033[0m dimensions.";
      LogErrorS(_error);


      std::cout << "\nTENSOR BATCHLESS DIMS:" << "\n";
      PrintDims(batchless_dims);

      std::cout << "\nImage required dims: [" << width << ", " << height << ", " << channels << "]\n\n";
      
      return nullptr;
    }
    if (batch_idx > dims[1])
    {
      std::string _error = "Tried to load a pinned tensor on batch index " + std::to_string(batch_idx) + ", whereas this tensor batch size is " + std::to_string(dims[1]) + ".";
      LogErrorS(_error);
    }



    float *image_data_float = tensor->cpu_tensor_ptr;
    int idx_offset =  (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);

    //std::cout << "worker idx: " << worker_idx << ", batch idx: " << batch_idx << ", batch offset: " << idx_offset << "\n";
  
    // Loop through each pixel and convert to float between 0.0 and 1.0
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
          // Assuming unsigned char has 8 bits, scale by 1/255.0 to get a float value between 0.0 and 1.0
          image_data_float[idx_offset + c * (height * width) + y * width + x] = (float)image_data[(y * width + x) * channels + c] / 255.0f;
        }
      }
    }
    stbi_image_free(image_data);

    //std::cout << "returning float image" << "\n";
    return image_data_float;
    
  } else {
    std::string img_n = img_name;
    std::string _error = "Failed to open image: " + img_n + ".\n\n";
    LogErrorS(_error);
  }

  return nullptr;
}





extern "C" float * wload_img_resize(Scope_Struct *scope_struct, DT_tensor *tensor, char *img_name, int worker_idx, int batch_idx, int c, int h, int w)
{
  //std::cout << "LOADING IMAGE FOR: " << tensor->name <<  "\n";
  //std::cout << "Image: " << img_name <<  "\n";


  int width, height, channels;

  //std::cout << "GLOAD IMG, dims of " << img_name << "\n";
  
  unsigned char* image_data = stbi_load(img_name, &width, &height, &channels, 0);



  if(width!=w||height!=h)
  { 
    image_data = interpolate_img(image_data, height, width, h, w);
    width = w;
    height = h;
  }



  if (image_data) {
    
    std::vector<int> dims = tensor->dims;

    
    
    std::vector<int> workerless_dims = BatchLessDims(dims);
    int workerless_dims_prod = DimsProd(workerless_dims);

    std::vector<int> batchless_dims = BatchLessDims(workerless_dims);
    int batchless_dims_prod = DimsProd(batchless_dims);
    


    if (batchless_dims_prod < width*height*channels)
    {
      std::string t_n = tensor->name;
      std::string _error = "The image dimensions are incompatible with the tensor \033[95m" + t_n + "\033[0m dimensions.";
      LogErrorS(_error);


      std::cout << "\nTENSOR BATCHLESS DIMS:" << "\n";
      PrintDims(batchless_dims);

      std::cout << "\nImage required dims: [" << width << ", " << height << ", " << channels << "]\n\n";
      
      return nullptr;
    }
    if (batch_idx > dims[1])
    {
      std::string _error = "Tried to load a pinned tensor on batch index " + std::to_string(batch_idx) + ", whereas this tensor batch size is " + std::to_string(dims[1]) + ".";
      LogErrorS(_error);
    }



    float *image_data_float = tensor->cpu_tensor_ptr;
    int idx_offset = batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx;

    //std::cout << "worker idx: " << worker_idx << ", batch idx: " << batch_idx << ", batch offset: " << idx_offset << "\n";
  
    // Loop through each pixel and convert to float between 0.0 and 1.0
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
          // Assuming unsigned char has 8 bits, scale by 1/255.0 to get a float value between 0.0 and 1.0
          image_data_float[idx_offset + c * (height * width) + y * width + x] = (float)image_data[(y * width + x) * channels + c] / 255.0f;
        }
      }
    }
    stbi_image_free(image_data);

    //std::cout << "returning float image" << "\n";
    return image_data_float;
    
  } else {
    std::string img_n = img_name;
    std::string _error = "Failed to open image: " + img_n + ".\n\n";
    LogErrorS(_error);
  }

  return nullptr;
}



float *current_data;
extern "C" float load_preprocess_img(Scope_Struct *scope_struct,DT_tensor tensor, char *img_name)
{
  float *img;
  img = load_img(scope_struct, img_name); 
  
  std::vector<int> dims = tensor.dims;

  
  // Last three dims.
  int img_dims_prod = dims[dims.size()-1]*dims[dims.size()-2]*dims[dims.size()-3];


  current_data = new float[img_dims_prod];
  cudaCheck(cudaMallocHost(&current_data, img_dims_prod*sizeof(float)));


  for (int j = 0; j < img_dims_prod; ++j)
    current_data[j] = img[j];
  delete[] img;


  float *x = tensor.tensor_ptr;
  //cudaMalloc(&x, img_dims_prod*sizeof(float));
  cudaCheck(cudaMemcpy(x, current_data, img_dims_prod*sizeof(float), cudaMemcpyHostToDevice));


  return 0;
}