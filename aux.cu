#include <cudnn.h>
#include <algorithm>
#include <cstdarg>
#include <cassert>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>
#include <iomanip>
#include <math.h>
#include <fenv.h>
#include <tuple>
#include <glob.h>
#include <chrono>
#include <thread>
#include <random>


// Cuda
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <omp.h>


/*
__global__ void sum_dims_kernel(const float *tensor,
                           float *summed,
                           int dims_prod, int summed_dim_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int C = summed_dim_size;
    
    if (i < dims_prod) {
        int b = i / (C);
        int v = i % C;

        float *summed_b = summed + b * C;

        int ix = tensor[b];


        float indicator = (v==ix) ? 1.0f : 0.0f;

        summed_b[v] = indicator;
    }
}
*/

int main(){

    const float *tensor;
    float *summed;

    int dims_prod = 40;
    int C = 5;
    int D = 2;
    
    for (int i=0; i<dims_prod; i++)
    {
        if (i < dims_prod) {
            int b = i / C;
            int d = i / D;
            int v = i % C;

            std::cout << "i: " << i << "\nb: " << b << "\nv: " << v << "\nd:" << d << "\nb*C+v: " << b*C+v << "\n\n";
            
            //float *summed_b = summed + b * C;

            //int ix = tensor[b];


            //float indicator = (v==ix) ? 1.0f : 0.0f;

            //summed_b[v] = indicator;
        }
    }

    return 0;
}