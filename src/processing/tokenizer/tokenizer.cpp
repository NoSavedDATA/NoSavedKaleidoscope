#include <algorithm>
#include <fstream>      // std::stringstream
#include <iostream>     // std::cout
#include <sstream>      // std::stringstream
#include <string>       // std::string
#include <string_view>       // std::string
#include <map> 
#include <vector>


#include "../../compiler_frontend/logging.h"
#include "../../mangler/scope_struct.h"
#include "../../simd/include.h"
#include "../../tensor/tensor_dim_functions.h"
#include "../../tensor/tensor_struct.h"
#include "../../threads/include.h"


std::unordered_map<std::string, int> Vocab;
int max_tokens;
int last_tok_id = 2;

int UNK_TOK = 1.0f;
int PAD_TOK = 0.0f;

inline void ProcessString(std::string& str) {
  // to lower and remove ponctuation


  // Convert to lowercase
  std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c) { return std::tolower(c); });

  // Remove punctuation
  str.erase(std::remove_if(str.begin(), str.end(),
                   [](unsigned char c) { return std::ispunct(c); }),
            str.end());
}

inline void ProcessString(std::string_view in, std::string& out) {
    out.clear();
    out.reserve(in.size());  // worst case: no characters removed

    for (char c : in) {
        if (!std::ispunct(static_cast<unsigned char>(c))) {
            out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
}

inline std::string_view ProcessStringToBuffer(std::string_view in, char* buffer) {
    size_t len = 0;
    for (char c : in) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (!std::ispunct(uc)) {
            buffer[len++] = std::tolower(uc);
        }
    }
    return std::string_view(buffer, len);
}





extern "C" float build_vocab(Scope_Struct *scope_struct, char *filename, int _max_tokens)
{
    max_tokens = _max_tokens;
    last_tok_id = 0;

    FILE* f = fopen(filename, "r");
    if (!f) {
        std::cerr << "Error opening file " << filename << std::endl;
        return 1;
    }

    constexpr size_t LINE_BUF_SIZE = 23*40;
    constexpr size_t WORD_BUF_SIZE = 23;
    char line_buffer[LINE_BUF_SIZE];
    char cleaned_buffer[WORD_BUF_SIZE];

    std::string stored_token;  // Used to insert into Vocab
    std::string processed_word;

    while (fgets(line_buffer, LINE_BUF_SIZE, f)) {
        char* token = std::strtok(line_buffer, " \t\n\r");
        while (token) {
            std::string_view word(token);
            // std::string_view cleaned = ProcessStringToBuffer(token, cleaned_buffer);
            ProcessString(word, processed_word);

            // Only copy and insert if it's a new token
            if (Vocab.find(processed_word) == Vocab.end() && last_tok_id < (max_tokens - 2)) {
                stored_token.assign(processed_word);  // copies cleaned string into std::string
                Vocab[stored_token] = last_tok_id++;
            }

            token = std::strtok(nullptr, " \t\n\r");
        }
    }

    fclose(f);
    return 0;
}



// extern "C" float tokenize(Scope_Struct *scope_struct, DT_tensor *tensor, char *filename)
// {
//   pthread_mutex_lock(&vocab_mutex); // Files are not thread safe
//   std::ifstream file(filename);

//   if (!file) {
//     std::cerr << "Error opening file!" << std::endl;
//     return 1;
//   }

//   std::string line;
//   while (std::getline(file, line)) 
//   {       
//     std::istringstream lineStream(line); // Create a string stream from the line
//     std::string word;

//     while (lineStream >> word) {

//       ProcessString(word);

//       int idx;

//       std::cout << word << "\n";
//       idx = (Vocab.count(word)>0) ? Vocab[word] : UNK_TOK;
      

//       std::cout << idx << std::endl;
//     }
//   }

//   file.close();
//   pthread_mutex_unlock(&vocab_mutex);

//   return 0;
// }


// extern "C" float wtokenize(Scope_Struct *scope_struct, DT_tensor *tensor, char *filename, int trunc_to, int worker_idx, int batch_idx)
// {
//   //tensor e [workers, seq_len, batch_size, vocab_size]

//   //pthread_mutex_lock(&vocab_mutex); // Files are not trhead safe
//   std::ifstream file(filename);

//   //std::cout << "Loading" <<  filename << "\n";
//   if (!file) {
//     std::cerr << "Error opening file!" << std::endl;
//     return 1;
//   }


//   std::vector<int> dims = tensor->dims;
//   int dims_prod = tensor->dims_prod;


//   std::vector<int> workerless_dims = BatchLessDims(dims);
//   int workerless_dims_prod = DimsProd(workerless_dims);

//   std::vector<int> seqless_dims = BatchLessDims(workerless_dims);
//   int seqless_dims_prod = DimsProd(seqless_dims);

//   std::vector<int> batchless_dims = BatchLessDims(seqless_dims);
//   int batchless_dims_prod = DimsProd(batchless_dims);

//   int idx_offset = (int) (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);


//   //TODO: add pad and left

//   std::string line;
//   int words_count = 0;
//   while (std::getline(file, line)) 
//   {       
//     std::istringstream lineStream(line); // Create a string stream from the line
//     std::string word;

//     while (lineStream >> word)
//     {
//       words_count++;
//       if (words_count>trunc_to)
//         break;

//       ProcessString(word);

//       int idx, pre_idx;

//       idx = (Vocab.count(word)>0) ? Vocab[word] : UNK_TOK; // inplace onehot
//       idx = (idx<0) ? 0 : idx;


//       pre_idx = idx;
//       idx = idx + idx_offset;

//       if(idx>dims_prod)
//       {
//         std::string _err = "Tokenizer IDX " + std::to_string(idx) + " is higher than allowed dims: " +std::to_string(dims_prod) + " from pre-idx: " + std::to_string(pre_idx);
//         LogErrorS(_err);
//       }

//       tensor->cpu_tensor_ptr[idx] = 1;

//       idx_offset += seqless_dims_prod; //moves to the next sequence element
//       /*
//       if(idx_offset>dims_prod)
//       {
//         std::string _err = "Tokenizer Index " + std::to_string(idx_offset) + " is higher than allowed dims: " +std::to_string(dims_prod);
//         LogErrorS(_err);
//       }
//       */
//     }
//     if (words_count>trunc_to)
//       break;
//   }

//   file.close();
//   //pthread_mutex_unlock(&vocab_mutex);

//   return 0;
// }


// extern "C" float wtokenize_pad_left(Scope_Struct *scope_struct, DT_tensor *tensor, char *filename, int trunc_to, int worker_idx, int batch_idx)
// {
//   // x e [W, T, B]
  
//   std::ifstream file(filename);

//   //std::cout << "Loading" <<  filename << "\n";
//   if (!file) {
//     std::cerr << "Error opening file!" << std::endl;
//     return 1;
//   }


//   std::vector<int> dims = tensor->dims;
//   int dims_prod = tensor->dims_prod;


//   std::vector<int> workerless_dims = BatchLessDims(dims);
//   int workerless_dims_prod = DimsProd(workerless_dims);

//   std::vector<int> seqless_dims = BatchLessDims(workerless_dims);
//   int seqless_dims_prod = DimsProd(seqless_dims);

//   std::vector<int> batchless_dims = BatchLessDims(seqless_dims);
//   int batchless_dims_prod = DimsProd(batchless_dims);

//   int idx_offset = (int) (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);



//   int *indices  = new int[trunc_to];
//   int *padded_indices  = new int[trunc_to];

//   //for (int i=0; i<tensor->dims_prod; i++)
//   //  tensor->cpu_tensor_ptr[i] = 0.0f;

//   for (int i = 0; i < trunc_to; i++)
//     padded_indices[i] = PAD_TOK;


//   std::string line;
//   int words_count = 0;
//   while (std::getline(file, line)) 
//   {       
//     std::istringstream lineStream(line); // Create a string stream from the line
//     std::string word;

//     while (lineStream >> word)
//     {
//       words_count++;
//       if (words_count>trunc_to)
//         break;

//       ProcessString(word);

//       int idx;

//       idx = (Vocab.count(word)>0) ? Vocab[word] : UNK_TOK; // inplace onehot
//       idx = (idx<0) ? 0 : idx;

//       indices[words_count-1] = idx;      
//     }
//     if (words_count>trunc_to)
//       break;
//   }
//   file.close();


//   if (words_count<=trunc_to)
//   {
//     int offset = trunc_to-words_count;
//     for(int i=0; i<words_count; i++)
//     {
//       padded_indices[i+offset] = indices[i];
//     }
//   } else 
//     padded_indices = indices;


//   int idx;
//   for(int i=0; i<trunc_to; i++)
//   {
//     idx = padded_indices[i] + idx_offset;

//     tensor->cpu_tensor_ptr[idx] = 1;

//     idx_offset += seqless_dims_prod; //moves to the next sequence element
//   }
  
//   /*
//   std::cout << "[";
//   for (int i = 0; i < trunc_to; i++)
//     std::cout << padded_indices[i] << ", ";
//   std::cout << "]" << "\n\n";  
//   */
  

//   delete[] indices;
//   if (words_count<=trunc_to)
//     delete[] padded_indices;

//   return 0;
// }



// extern "C" float wtokenize_pad_left_batch_first(Scope_Struct *scope_struct, DT_tensor *tensor, char *filename, int trunc_to, int worker_idx, int batch_idx)
// {
//   // x e [W, B, T, V]
  

//   std::ifstream file(filename);

//   //std::cout << "Loading" <<  filename << "\n";
//   if (!file) {
//     std::cerr << "Error opening file!" << std::endl;
//     return 1;
//   }


//   std::vector<int> dims = tensor->dims;
//   int dims_prod = tensor->dims_prod;


//   std::vector<int> workerless_dims = BatchLessDims(dims);
//   int workerless_dims_prod = DimsProd(workerless_dims);

//   std::vector<int> batchless_dims = BatchLessDims(workerless_dims);
//   int batchless_dims_prod = DimsProd(batchless_dims);

//   std::vector<int> seqless_dims = BatchLessDims(batchless_dims);
//   int seqless_dims_prod = DimsProd(seqless_dims);



//   int idx_offset = (int) (batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx);



//   int *indices  = new int[trunc_to];
//   int *padded_indices  = new int[trunc_to];

//   //for (int i=0; i<tensor->dims_prod; i++)
//   //  tensor->cpu_tensor_ptr[i] = 0.0f;

//   for (int i = 0; i < trunc_to; i++)
//     padded_indices[i] = PAD_TOK;


//   std::string line;
//   int words_count = 0;
//   while (std::getline(file, line)) 
//   {       
//     std::istringstream lineStream(line); // Create a string stream from the line
//     std::string word;

//     while (lineStream >> word)
//     {
//       words_count++;
//       if (words_count>trunc_to)
//         break;

//       ProcessString(word);

//       int idx;

//       idx = (Vocab.count(word)>0) ? Vocab[word] : UNK_TOK; // inplace onehot
//       idx = (idx<0) ? 0 : idx;

//       indices[words_count-1] = idx;      
//     }
//     if (words_count>trunc_to)
//       break;
//   }
//   file.close();



//   // pad indices
//   if (words_count<=trunc_to)
//   {
//     int offset = trunc_to-words_count;
//     for(int i=0; i<words_count; i++)
//     {
//       padded_indices[i+offset] = indices[i];
//     }
//   } else 
//     padded_indices = indices;



//   // one-hot and save it into the tensor
//   int idx;
//   for(int i=0; i<trunc_to; i++)
//   {
//     idx = padded_indices[i] + idx_offset;

//     tensor->cpu_tensor_ptr[idx] = 1;

//     idx_offset += seqless_dims_prod; //moves to the next sequence element
//   }

  
  
//   delete[] indices;
//   if (words_count<=trunc_to)
//     delete[] padded_indices;

//   return 0;
// }


extern "C" float wtokenize_pad_left_idx(Scope_Struct *scope_struct, DT_tensor *tensor, char *filename, int trunc_to, int worker_idx, int batch_idx)
{
  // x e [W, B, T]
  

  // std::ifstream file(filename);
  // if (!file) {
  //   std::cerr << "Error opening file!" << std::endl;
  //   return 1;
  // }
  FILE* file = std::fopen(filename, "r");
  if (!file) {
      std::cerr << "Error opening file!" << std::endl;
      return 1;
  }


  // std::cout << "Loading" <<  filename << "\n";
  // std::cout << "trunc to " << trunc_to << " worker " << worker_idx << " batch_idx " << batch_idx << ".\n";


  std::vector<int> dims = tensor->dims;
  int dims_prod = tensor->dims_prod;


  std::vector<int> workerless_dims = BatchLessDims(dims);
  int workerless_dims_prod = DimsProd(workerless_dims);

  std::vector<int> batchless_dims = BatchLessDims(workerless_dims);
  int batchless_dims_prod = DimsProd(batchless_dims);

  


  int idx_offset = batchless_dims_prod*batch_idx + workerless_dims_prod*worker_idx;


  // float *indices  = new float[trunc_to];
  // float *padded_indices = new float[trunc_to];

  float *indices = (float*)std::aligned_alloc(32, trunc_to * sizeof(float));
  float *padded_indices = (float*)std::aligned_alloc(32, trunc_to * sizeof(float));


  //for (int i=0; i<tensor->dims_prod; i++)
  //  tensor->cpu_tensor_ptr[i] = 0.0f;

  // for (int i = 0; i < trunc_to; i++)
  //   padded_indices[i] = (float)PAD_TOK;
  // memset(padded_indices, 0, trunc_to * sizeof(float)); // Sets all bytes to 0 = 0.0f
  simd_fill_float_aligned(padded_indices, 0, trunc_to);



  int words_count = 0;
  
  std::string processed_word;
  constexpr size_t LINE_BUF_SIZE = 23*30;  // Adjust if your lines are longer
  char line_buffer[LINE_BUF_SIZE];

  constexpr int MAX_TOKEN_LEN = 23;
  char cleaned_buf[MAX_TOKEN_LEN];


  while (std::fgets(line_buffer, LINE_BUF_SIZE, file)) {
    char *saveptr;
    char* token = strtok_r(line_buffer, " \t\n\r", &saveptr);  // split on whitespace

    // while (lineStream >> word)
    while(token && words_count<=trunc_to)
    {
      std::string_view word(token);
      
      ProcessString(word, processed_word);
      // word = ProcessStringToBuffer(word, cleaned_buf)
      // std::cout << "processed word " << processed_word << ".\n";
      // std::cout << "view word " << processed_word << ".\n";

      auto it = Vocab.find(processed_word);
      int idx = (it != Vocab.end()) ? std::max(0, it->second) : UNK_TOK;

      indices[words_count++] = (float)idx;
      token = strtok_r(nullptr, " \t\n\r", &saveptr);
    }

    if (words_count>trunc_to)
      break;
  }
  // file.close();
  std::fclose(file);




  // pad indices
  if (words_count<=trunc_to)
  {
    int offset = trunc_to-words_count;
    for(int i=0; i<words_count; i++)
      padded_indices[i+offset] = (float)indices[i];

  } else {
    free(padded_indices);
    padded_indices = indices;
  }




  int idx = idx_offset;

  // memcpy(tensor->cpu_tensor_ptr + idx_offset, padded_indices, trunc_to * sizeof(float));
  simd_copy_floats(tensor->cpu_tensor_ptr + idx_offset, padded_indices, trunc_to);


  // for(int i=0; i<trunc_to; i++)
  // {
  //   tensor->cpu_tensor_ptr[idx] = padded_indices[i];


  //   idx += 1; //moves to the next sequence element
  // }
  
  
  free(indices);
  if (words_count<=trunc_to)
    free(padded_indices);

  return 0;
}