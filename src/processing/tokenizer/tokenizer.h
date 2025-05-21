#pragma once

#include <map>

#include "../../tensor/tensor_struct.h"


extern std::unordered_map<std::string_view, int> Vocab;
extern std::vector<std::string> VocabStorage;

extern int max_tokens;
extern int last_tok_id;

extern int UNK_TOK;
extern int PAD_TOK;


void ProcessString(std::string& str);


// extern "C" float build_vocab(char *filename, float _max_tokens);



// extern "C" float tokenize(DT_tensor *tensor, char *filename);



// extern "C" float wtokenize(DT_tensor *tensor, char *filename, float trunc_to, float worker_idx, float batch_idx);



// extern "C" float wtokenize_pad_left(DT_tensor *tensor, char *filename, float trunc_to, float worker_idx, float batch_idx);




// extern "C" float wtokenize_pad_left_batch_first(DT_tensor *tensor, char *filename, float trunc_to, float worker_idx, float batch_idx);



// extern "C" float wtokenize_pad_left_idx(DT_tensor *tensor, char *filename, float trunc_to, float worker_idx, float batch_idx);
