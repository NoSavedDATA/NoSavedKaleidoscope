#pragma once


#include <map>
#include <string>
#include <memory>


#include "BatchNorm2d/class.h"
#include "Conv2d/class.h"
#include "Embedding/class.h"
#include "Linear/class.h"
#include "LSTM/class.h"
#include "MaxPool2d/class.h"
#include "MHSA/class.h"

extern std::map<std::string, std::unique_ptr<BatchNorm2dCPP>> NamedBatchNorm2d;
extern std::map<std::string, std::unique_ptr<Conv2dCPP>> NamedConv2d;
extern std::map<std::string, std::unique_ptr<Embedding>> NamedEmbedding;
extern std::map<std::string, std::unique_ptr<LinearCPP>> NamedLinear;
extern std::map<std::string, std::unique_ptr<LSTM>> NamedLSTM;
extern std::map<std::string, std::unique_ptr<MaxPool2dCPP>> NamedMaxPool2d;
extern std::map<std::string, std::unique_ptr<MHSA>> NamedMHSA;