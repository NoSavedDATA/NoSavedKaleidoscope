#!/usr/bin/env bash
set -e

# Ensure submodules in include/ are initialized and updated
git submodule update --init --recursive include

# Enter include/ and pull the latest commits for each submodule
cd include

git submodule foreach '
    echo "Updating submodule: $name"
    # Try to checkout main (ignore error if branch name differs)
    git checkout main 2>/dev/null || true
    # Pull latest changes from origin
    git pull origin main 2>/dev/null || true
'

cd ..
echo "✅ All include/ submodules updated."

