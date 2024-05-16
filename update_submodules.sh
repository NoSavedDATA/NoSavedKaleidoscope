#!/bin/bash

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "This script must be run from within a git repository."
    exit 1
fi

# Initialize and update git submodules
echo "Initializing and updating git submodules..."
git submodule update --init --recursive

if [ $? -eq 0 ]; then
    echo "Submodules have been successfully initialized and updated."
else
    echo "There was an error initializing or updating submodules."
    exit 1
fi
