#pragma once

#include <map>

#include "../../tensor/tensor_struct.h"


extern std::map<std::string, int> Vocab;
extern float max_tokens;
extern float last_tok_id;

extern int UNK_TOK;
extern int PAD_TOK;


void ProcessString(std::string& str);


extern "C" float build_vocab(char *filename, float _max_tokens);



extern "C" float tokenize(data_type_tensor *tensor, char *filename);



extern "C" float wtokenize(data_type_tensor *tensor, char *filename, float trunc_to, float worker_idx, float batch_idx);



extern "C" float wtokenize_pad_left(data_type_tensor *tensor, char *filename, float trunc_to, float worker_idx, float batch_idx);




extern "C" float wtokenize_pad_left_batch_first(data_type_tensor *tensor, char *filename, float trunc_to, float worker_idx, float batch_idx);



extern "C" float wtokenize_pad_left_idx(data_type_tensor *tensor, char *filename, float trunc_to, float worker_idx, float batch_idx);
