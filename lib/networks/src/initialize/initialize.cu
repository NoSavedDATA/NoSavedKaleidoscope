#include "../../../nsk_cuda/include.h"

#include "../modules/include.h"


extern "C" void initialize__networks() { 
    backward_functions["linear_backward"] = linear_backward;
	backward_functions["mhsa_backward"] = mhsa_backward;
	backward_functions["embedding_backward"] = embedding_backward;
	backward_functions["batchnorm2d_backward"] = batchnorm2d_backward;
	backward_functions["lstm_backward"] = lstm_backward;
	backward_functions["pool2d_backward"] = pool2d_backward;
    backward_functions["conv2d_backward"] = conv2d_backward;
	backward_functions["embeddingln_backward"] = embeddingln_backward;

}