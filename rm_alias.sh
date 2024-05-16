#!/bin/bash

remove_alias() {
    unset -f nsk
    unset -f cnsk
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "$DIR/bin/nsk" | tr '\n' ':' | sed 's/:$//' )

remove_alias
