#!/bin/bash

remove_alias() {
    if [[ $SHELL == *"zsh"* ]]; then
        unalias nsk 2>/dev/null
        unalias cnsk 2>/dev/null
    else
        unset -f nsk
        unset -f cnsk
    fi
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "$DIR/bin/nsk" | tr '\n' ':' | sed 's/:$//' )

remove_alias
