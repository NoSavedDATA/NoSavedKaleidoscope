#pragma once

#include <string>

struct DT_placeholder {
    std::string name;
    int x, y;

    DT_placeholder(char *name, int x, int y);
};