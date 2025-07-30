
#include "../../../../src/nsk_cpp.h"
#include "../common/cu_commons.h"
#include "../backprop/backprop.h"


extern "C" float eval(Scope_Struct *scope_struct)
{
  std::cout << "\n\n\nSETTING NN MODE TO EVAL" << "\n\n";
    
  for (auto& pair : NamedParamGrads)
  {
    std::cout << "Erasing gradient memory of: " << pair.first << "\n";
    cudaCheck(cudaFree(pair.second));
  }

  NamedParamGrads.clear();



  // Todo: clean optimizer grads
  // for (auto &pair: optimizer->NamedV)
  //   cudaCheck(cudaFree(pair.second));
    
  // for (auto &pair: optimizer->NamedM)
  //   cudaCheck(cudaFree(pair.second));  

  nn_mode = eval_mode;

  std::cout << "\n\n\n";
  return 0;
}




extern "C" float train(Scope_Struct *scope_struct)
{
  std::cout << "SETTING NN MODE TO TRAIN" << "\n\n";
  nn_mode = training_mode;
  return 0;
}