#!/bin/bash

remove_culang_alias() {
    unset -f culang
    unset -f cumpile
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "$DIR/bin/culang" | tr '\n' ':' | sed 's/:$//' )

remove_culang_alias
