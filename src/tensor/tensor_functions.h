
#pragma once


extern "C" float CreateTensorOnDemand(char *tensor_name, char *scopeless_name, char *init, int is_weight, int thread_id, char *scope);
