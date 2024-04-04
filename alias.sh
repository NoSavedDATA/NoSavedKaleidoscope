#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if ! command -v culang &> /dev/null; then
    echo "Adding culang to PATH"
    export PATH="$DIR/bin/culang:$PATH"
fi

cumpile() {
    clang++ -g -O3 -rdynamic toy.cu `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` --cuda-path="/usr/local/cuda-12" --cuda-gpu-arch=sm_75 -L"/usr/local/cuda-12/lib64" -lcudart_static -ldl -lrt -pthread -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -o bin/culang
}

export -f cumpile
