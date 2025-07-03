#!/bin/bash

# Define the alias command

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

alias_command="alias nsk='$DIR/bin/nsk'"
nsk_lib="export NSK_LIBS='$DIR/lib'"


# Check if the alias already exists in the shell configuration file
if ! grep -q "$alias_command" ~/.bashrc; then
    # If the alias doesn't exist, append it to the shell configuration file
    echo "$alias_command" >> ~/.bashrc
    echo "Alias 'nsk' created successfully."
else
    echo "Alias 'nsk' already exists."
fi


if ! grep -q "$nsk_lib" ~/.bashrc; then
    # If the alias doesn't exist, append it to the shell configuration file
    echo "$nsk_lib" >> ~/.bashrc
fi


#alias_command="alias compile_nsk='clang++ -g -O3 -rdynamic $DIR/toy.cu `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native` --cuda-path=/usr/local/cuda-12 --cuda-gpu-arch=sm_75 -L/usr/local/cuda-12/lib64 -lcudart_static  -lcublas -lcublasLt -ldl -lrt -pthread -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -w -o bin/nsk'"


#if ! grep -q "$alias_command" ~/.bashrc; then
#    echo "$alias_command" >> ~/.bashrc
#    echo "Alias 'compile_nsk' created successfully."
#else
#    echo "Alias 'compile_nsk' already exists."
#fi






# Source the modified shell configuration file to apply the changes
exec bash