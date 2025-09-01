#pragma once



#include <iostream>
#include <map>
#include <string>

#include "../data_types/list.h"
#include "barrier.h"




struct Channel {

    DT_list *data_list;
    std::string name;

    Channel();
};