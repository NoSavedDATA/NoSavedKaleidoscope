#pragma once






__device__ void my_device_func() {

    // std::cout << "OPA OPA OPA" << ".\n";
    printf("blockIdx.x = %d, threadIdx.x = %d oo\n", blockIdx.x, threadIdx.x);
}


