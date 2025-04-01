#pragma once


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

extern std::map<std::string, std::unique_ptr<BatchNorm2d>> NamedBatchNorm2d;
extern std::map<std::string, std::unique_ptr<BN2dRelu>> NamedBN2dRelu;
extern std::map<std::string, std::unique_ptr<Conv2d>> NamedConv2d;
extern std::map<std::string, std::unique_ptr<Embedding>> NamedEmbedding;
extern std::map<std::string, std::unique_ptr<Linear>> NamedLinear;
extern std::map<std::string, std::unique_ptr<LSTM>> NamedLSTM;
extern std::map<std::string, std::unique_ptr<MaxPool2d>> NamedMaxPool2d;
extern std::map<std::string, std::unique_ptr<MHSA>> NamedMHSA;
extern std::map<std::string, std::unique_ptr<Relu>> NamedRelu;