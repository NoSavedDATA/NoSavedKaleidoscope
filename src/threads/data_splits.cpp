#include <iostream>
#include <cmath>

#include "../mangler/scope_struct.h"




extern "C" std::vector<float> *float_vec_Split_Parallel(Scope_Struct *scope_struct, std::vector<float> vec)
{
    float threads_count = scope_struct->asyncs_count;
    float thread_id = scope_struct->thread_id-1;
    std::cout << "SPLITTING FLOAT VEC"  << ".\n";
    std::cout << "Threads count: " << scope_struct->asyncs_count << ".\n";
    std::cout << "Threads id: " << scope_struct->thread_id << ".\n\n";

    int vec_size = vec.size();
    int segment_size;

    segment_size = ceilf(vec_size/threads_count);

    std::cout << "SEGMENT SIZE IS " << segment_size << ".\n";

    
    std::vector<float> *out_vector = new std::vector<float>();


    for (int i = segment_size*thread_id; i<segment_size*(thread_id+1) && i<vec.size(); ++i)
        out_vector->push_back(vec[i]);

    return out_vector;
}
