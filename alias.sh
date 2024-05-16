#!/bin/bash

# Define the alias commands
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

alias_command="alias nsk='$DIR/bin/nsk'"
alias_compile="alias cnsk='clang++ -g -O3 -rdynamic $DIR/toy.cu \`llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native\` --cuda-path=/usr/local/cuda-12 --cuda-gpu-arch=sm_75 -L/usr/local/cuda-12/lib64 -lcudart_static -lcublas -lcublasLt -ldl -lrt -pthread -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -w -o bin/nsk'"

# Function to add alias to shell configuration file
add_alias() {
    local shell_config=$1
    local alias_command=$2

    if ! grep -q "$alias_command" "$shell_config"; then
        echo "$alias_command" >> "$shell_config"
        echo "Alias added to $shell_config"
    else
        echo "Alias already exists in $shell_config"
    fi
}

# Detect shell and update appropriate configuration file
if [ -n "$ZSH_VERSION" ]; then
    shell_config="$HOME/.zshrc"
    shell_name="zsh"
else
    shell_config="$HOME/.bashrc"
    shell_name="bash"
fi

# Add the aliases
add_alias "$shell_config" "$alias_command"
add_alias "$shell_config" "$alias_compile"

# Source the modified shell configuration file to apply the changes
if [ "$shell_name" = "zsh" ]; then
    source "$shell_config"
    echo "Restarting zsh to apply changes..."
    exec zsh
else
    source "$shell_config"
    echo "Restarting bash to apply changes..."
    exec bash
fi
