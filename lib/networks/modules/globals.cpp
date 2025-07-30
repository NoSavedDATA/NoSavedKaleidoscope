#include <map>
#include <string>
#include <memory>


#include "BatchNorm2d/class.h"
#include "Embedding/class.h"
#include "EmbeddingLn/class.h"
#include "Linear/class.h"
#include "LSTM/class.h"
#include "MaxPool2d/class.h"
#include "MHSA/class.h"


std::map<std::string, std::unique_ptr<BatchNorm2dCPP>> NamedBatchNorm2d;
std::map<std::string, std::unique_ptr<DT_Embedding>> NamedEmbedding;
std::map<std::string, std::unique_ptr<DT_EmbeddingLn>> NamedEmbeddingLn;
std::map<std::string, std::unique_ptr<LinearCPP>> NamedLinear;
std::map<std::string, std::unique_ptr<DT_LSTM>> NamedLSTM;
std::map<std::string, std::unique_ptr<MaxPool2dCPP>> NamedMaxPool2d;
std::map<std::string, std::unique_ptr<MHSA>> NamedMHSA;

