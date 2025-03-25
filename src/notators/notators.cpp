#include "notators.h"

int_vec *CreateIntVec(int *vec, int size)
{

  int_vec *ivec = new int_vec();
  ivec->vec = vec;
  ivec->size = size;
  return ivec;
}

int_vec *SetNotators(std::vector<int> Notators)
{

  int_vec *ivec = new int_vec();

  int notators_size = Notators.size();
  int *notators = new int[notators_size];
  for (int i=0; i<notators_size; ++i)
    notators[i] = Notators[i];

  ivec->vec = notators;
  ivec->size = notators_size;
  return ivec;
}