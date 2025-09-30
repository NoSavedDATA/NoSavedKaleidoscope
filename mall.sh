#!/usr/bin/env bash
set -euo pipefail

# folders to build (in order)
dirs=( "lib/nsk_cuda" "lib/networks" "." )

for d in "${dirs[@]}"; do
  if [[ -d "$d" ]]; then
    printf '\n==> Building %s\n' "$d"
    pushd "$d" >/dev/null
    # run configure if present & executable
    if [[ -x ./configure ]]; then
      printf 'Running ./configure in %s\n' "$d"
      ./configure
    fi
    # run make if Makefile exists
    if [[ -f Makefile || -f makefile ]]; then
      make -j8
    else
      printf 'No Makefile in %s — skipping make\n' "$d"
    fi
    popd >/dev/null
  else
    printf 'Directory not found: %s — skipping\n' "$d"
  fi
done

