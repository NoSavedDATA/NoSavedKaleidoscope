
#include <map>
#include <string>
#include <memory>


#include "BatchNorm2d/class.h"
#include "Bn2dRelu/class.h"
#include "Conv2d/class.h"
#include "CudnnRelu/class.h"
#include "Embedding/class.h"
#include "Linear/class.h"
#include "LSTM/class.h"
#include "MaxPool2d/class.h"
#include "MHSA/class.h"


std::map<std::string, std::unique_ptr<BatchNorm2d>> NamedBatchNorm2d;
std::map<std::string, std::unique_ptr<BN2dRelu>> NamedBN2dRelu;
std::map<std::string, std::unique_ptr<Conv2d>> NamedConv2d;
std::map<std::string, std::unique_ptr<Embedding>> NamedEmbedding;
std::map<std::string, std::unique_ptr<Linear>> NamedLinear;
std::map<std::string, std::unique_ptr<LSTM>> NamedLSTM;
std::map<std::string, std::unique_ptr<MaxPool2d>> NamedMaxPool2d;
std::map<std::string, std::unique_ptr<MHSA>> NamedMHSA;
std::map<std::string, std::unique_ptr<Relu>> NamedRelu;
