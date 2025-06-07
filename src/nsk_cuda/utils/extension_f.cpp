
#include <algorithm>
#include <cuda_fp16.h>

bool in_float_ptr_vec(const float *value, const std::vector<float *>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}
bool in_half_ptr_vec(const half *value, const std::vector<half *>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}
bool in_int8_ptr_vec(const int8_t* value, const std::vector<int8_t*>& list) {
    return std::find(list.begin(), list.end(), value) != list.end();
}