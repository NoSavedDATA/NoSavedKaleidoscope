#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>      // std::stringstream
#include <fstream>      // std::stringstream
#include <map> 
#include <vector>
#include <algorithm>


#include "../../codegen/tensor_dim_functions.h"
#include "../../compiler_frontend/logging.h"
#include "../../tensor/tensor_struct.h"
#include "../../threads/include.h"


std::map<std::string, int> Vocab;
float max_tokens;
float last_tok_id = 2;

int UNK_TOK = 1.0f;
int PAD_TOK = 0.0f;

void ProcessString(std::string& str) {
  // to lower and remove ponctuation


  // Convert to lowercase
  std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c) { return std::tolower(c); });

  // Remove punctuation
  str.erase(std::remove_if(str.begin(), str.end(),
                   [](unsigned char c) { return std::ispunct(c); }),
            str.end());
}


extern "C" float build_vocab(char *filename, float _max_tokens)
{
  pthread_mutex_lock(&vocab_mutex); // Files are not thread safe
  std::ifstream file(filename);
  max_tokens = _max_tokens;

  if (!file) {
    std::cerr << "Error opening file " << filename << std::endl;
    return 1;
  }

  std::string line;
  while (std::getline(file, line)) 
  {       
    std::istringstream lineStream(line); // Create a string stream from the line
    std::string word;

    while (lineStream >> word) {
      ProcessString(word);
      
      if (Vocab.count(word)==0 && last_tok_id<(max_tokens-2)) // -2 for padding and unk tokens
      {
        Vocab[word] = last_tok_id;
        last_tok_id+=1;
      }
    }
  }

  file.close();
  pthread_mutex_unlock(&vocab_mutex);

  return 0;
}


extern "C" float tokenize(Tensor *tensor, char *filename)
{
  pthread_mutex_lock(&vocab_mutex); // Files are not thread safe
  std::ifstream file(filename);

  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return 1;
  }

  std::string line;
  while (std::getline(file, line)) 
  {       
    std::istringstream lineStream(line); // Create a string stream from the line
    std::string word;

    while (lineStream >> word) {

      ProcessString(word);

      int idx;

      std::cout << word << "\n";
      idx = (Vocab.count(word)>0) ? Vocab[word] : UNK_TOK;
      

      std::cout << idx << std::endl;
    }
  }

  file.close();
  pthread_mutex_unlock(&vocab_mutex);

  return 0;
}


extern "C" float wtokenize(Tensor *tensor, char *filename, float trunc_to, float worker_idx, float batch_idx)
{
  //tensor e [workers, seq_len, batch_size, vocab_size]

  //pthread_mutex_lock(&vocab_mutex); // Files are not trhead safe
  std::ifstream file(filename);

  //std::cout << "Loading" <<  filename << "\n";
  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return 1;
  }


  std::vector<float> dims = tensor->dims;
  float dims_prod = tensor->dims_prod;


  std::vector<float> workerless_dims = BatchLessDims(dims);
  int workerless_dims_prod = DimsProd(workerless_dims);

  std::vector<float> seqless_dims = BatchLessDims(workerless_dims);
  int seqless_dims_prod = DimsProd(seqless_dims);

  std::vector<float> batchless_dims = BatchLessDims(seqless_dims);
  int batchless_dims_prod = DimsProd(batchless_dims);

  int idx_offset = (int) (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);


  //TODO: add pad and left

  std::string line;
  int words_count = 0;
  while (std::getline(file, line)) 
  {       
    std::istringstream lineStream(line); // Create a string stream from the line
    std::string word;

    while (lineStream >> word)
    {
      words_count++;
      if (words_count>trunc_to)
        break;

      ProcessString(word);

      int idx, pre_idx;

      idx = (Vocab.count(word)>0) ? Vocab[word] : UNK_TOK; // inplace onehot
      idx = (idx<0) ? 0 : idx;


      pre_idx = idx;
      idx = idx + idx_offset;

      if(idx>dims_prod)
      {
        std::string _err = "Tokenizer IDX " + std::to_string(idx) + " is higher than allowed dims: " +std::to_string(dims_prod) + " from pre-idx: " + std::to_string(pre_idx);
        LogErrorS(_err);
      }

      tensor->cpu_tensor_ptr[idx] = 1;

      idx_offset += seqless_dims_prod; //moves to the next sequence element
      /*
      if(idx_offset>dims_prod)
      {
        std::string _err = "Tokenizer Index " + std::to_string(idx_offset) + " is higher than allowed dims: " +std::to_string(dims_prod);
        LogErrorS(_err);
      }
      */
    }
    if (words_count>trunc_to)
      break;
  }

  file.close();
  //pthread_mutex_unlock(&vocab_mutex);

  return 0;
}


extern "C" float wtokenize_pad_left(Tensor *tensor, char *filename, float trunc_to, float worker_idx, float batch_idx)
{
  // x e [W, T, B]
  
  std::ifstream file(filename);

  //std::cout << "Loading" <<  filename << "\n";
  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return 1;
  }


  std::vector<float> dims = tensor->dims;
  float dims_prod = tensor->dims_prod;


  std::vector<float> workerless_dims = BatchLessDims(dims);
  int workerless_dims_prod = DimsProd(workerless_dims);

  std::vector<float> seqless_dims = BatchLessDims(workerless_dims);
  int seqless_dims_prod = DimsProd(seqless_dims);

  std::vector<float> batchless_dims = BatchLessDims(seqless_dims);
  int batchless_dims_prod = DimsProd(batchless_dims);

  int idx_offset = (int) (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);



  int *indices  = new int[trunc_to];
  int *padded_indices  = new int[trunc_to];

  //for (int i=0; i<tensor->dims_prod; i++)
  //  tensor->cpu_tensor_ptr[i] = 0.0f;

  for (int i = 0; i < trunc_to; i++)
    padded_indices[i] = PAD_TOK;


  std::string line;
  int words_count = 0;
  while (std::getline(file, line)) 
  {       
    std::istringstream lineStream(line); // Create a string stream from the line
    std::string word;

    while (lineStream >> word)
    {
      words_count++;
      if (words_count>trunc_to)
        break;

      ProcessString(word);

      int idx;

      idx = (Vocab.count(word)>0) ? Vocab[word] : UNK_TOK; // inplace onehot
      idx = (idx<0) ? 0 : idx;

      indices[words_count-1] = idx;      
    }
    if (words_count>trunc_to)
      break;
  }
  file.close();


  if (words_count<=trunc_to)
  {
    int offset = trunc_to-words_count;
    for(int i=0; i<words_count; i++)
    {
      padded_indices[i+offset] = indices[i];
    }
  } else 
    padded_indices = indices;


  int idx;
  for(int i=0; i<trunc_to; i++)
  {
    idx = padded_indices[i] + idx_offset;

    tensor->cpu_tensor_ptr[idx] = 1;

    idx_offset += seqless_dims_prod; //moves to the next sequence element
  }
  
  /*
  std::cout << "[";
  for (int i = 0; i < trunc_to; i++)
    std::cout << padded_indices[i] << ", ";
  std::cout << "]" << "\n\n";  
  */
  

  delete[] indices;
  if (words_count<=trunc_to)
    delete[] padded_indices;

  return 0;
}



extern "C" float wtokenize_pad_left_batch_first(Tensor *tensor, char *filename, float trunc_to, float worker_idx, float batch_idx)
{
  // x e [W, B, T, V]
  

  std::ifstream file(filename);

  //std::cout << "Loading" <<  filename << "\n";
  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return 1;
  }


  std::vector<float> dims = tensor->dims;
  float dims_prod = tensor->dims_prod;


  std::vector<float> workerless_dims = BatchLessDims(dims);
  int workerless_dims_prod = DimsProd(workerless_dims);

  std::vector<float> batchless_dims = BatchLessDims(workerless_dims);
  int batchless_dims_prod = DimsProd(batchless_dims);

  std::vector<float> seqless_dims = BatchLessDims(batchless_dims);
  int seqless_dims_prod = DimsProd(seqless_dims);



  int idx_offset = (int) (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);



  int *indices  = new int[trunc_to];
  int *padded_indices  = new int[trunc_to];

  //for (int i=0; i<tensor->dims_prod; i++)
  //  tensor->cpu_tensor_ptr[i] = 0.0f;

  for (int i = 0; i < trunc_to; i++)
    padded_indices[i] = PAD_TOK;


  std::string line;
  int words_count = 0;
  while (std::getline(file, line)) 
  {       
    std::istringstream lineStream(line); // Create a string stream from the line
    std::string word;

    while (lineStream >> word)
    {
      words_count++;
      if (words_count>trunc_to)
        break;

      ProcessString(word);

      int idx;

      idx = (Vocab.count(word)>0) ? Vocab[word] : UNK_TOK; // inplace onehot
      idx = (idx<0) ? 0 : idx;

      indices[words_count-1] = idx;      
    }
    if (words_count>trunc_to)
      break;
  }
  file.close();



  // pad indices
  if (words_count<=trunc_to)
  {
    int offset = trunc_to-words_count;
    for(int i=0; i<words_count; i++)
    {
      padded_indices[i+offset] = indices[i];
    }
  } else 
    padded_indices = indices;



  // one-hot and save it into the tensor
  int idx;
  for(int i=0; i<trunc_to; i++)
  {
    idx = padded_indices[i] + idx_offset;

    tensor->cpu_tensor_ptr[idx] = 1;

    idx_offset += seqless_dims_prod; //moves to the next sequence element
  }

  
  
  delete[] indices;
  if (words_count<=trunc_to)
    delete[] padded_indices;

  return 0;
}


extern "C" float wtokenize_pad_left_idx(Tensor *tensor, char *filename, float trunc_to, float worker_idx, float batch_idx)
{
  // x e [W, B, T]
  

  std::ifstream file(filename);

  //std::cout << "Loading" <<  filename << "\n";
  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return 1;
  }


  std::vector<float> dims = tensor->dims;
  float dims_prod = tensor->dims_prod;


  std::vector<float> workerless_dims = BatchLessDims(dims);
  int workerless_dims_prod = DimsProd(workerless_dims);

  std::vector<float> batchless_dims = BatchLessDims(workerless_dims);
  int batchless_dims_prod = DimsProd(batchless_dims);

  



  int idx_offset = (int) (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);



  int *indices  = new int[trunc_to];
  int *padded_indices  = new int[trunc_to];

  //for (int i=0; i<tensor->dims_prod; i++)
  //  tensor->cpu_tensor_ptr[i] = 0.0f;

  for (int i = 0; i < trunc_to; i++)
    padded_indices[i] = PAD_TOK;


  std::string line;
  int words_count = 0;
  while (std::getline(file, line)) 
  {       
    std::istringstream lineStream(line); // Create a string stream from the line
    std::string word;

    while (lineStream >> word)
    {
      words_count++;
      if (words_count>trunc_to)
        break;

      ProcessString(word);

      int idx;

      idx = (Vocab.count(word)>0) ? Vocab[word] : UNK_TOK; // inplace onehot
      idx = (idx<0) ? 0 : idx;

      indices[words_count-1] = idx;      
    }
    if (words_count>trunc_to)
      break;
  }
  file.close();



  // pad indices
  if (words_count<=trunc_to)
  {
    int offset = trunc_to-words_count;
    for(int i=0; i<words_count; i++)
    {
      padded_indices[i+offset] = indices[i];
    }
  } else 
    padded_indices = indices;




  //std::cout << "[";
  // one-hot and save it into the tensor
  int idx = idx_offset;
  for(int i=0; i<trunc_to; i++)
  {
    tensor->cpu_tensor_ptr[idx] = padded_indices[i];

    //std::cout << padded_indices[i] << ",";

    idx += 1; //moves to the next sequence element
  }
  //std::cout << "]\n";
  
  
  delete[] indices;
  if (words_count<=trunc_to)
    delete[] padded_indices;

  return 0;
}