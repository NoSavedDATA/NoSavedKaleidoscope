#pragma once

#include <vector>
#include <string>
#include <map>

enum Notators {
    bias=0,
    fp32=1,
    fp16=2,
    causal=3,
  };

struct int_vec{
  int *vec;
  int size;
};


extern std::map<std::string, int> NotatorsMap;


int_vec *CreateIntVec(int *vec, int size);

int_vec *SetNotators(std::vector<int> Notators);
