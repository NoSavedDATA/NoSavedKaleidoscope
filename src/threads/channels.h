#pragma once


#include <condition_variable>
#include <iostream>
#include <map>
#include <mutex>
#include <string>


#include "../data_types/list.h"
#include "barrier.h"




struct Channel {

    int buffer_size;
    bool terminated=false;

    DT_list *data_list;
    
    
    std::mutex mtx;
    std::condition_variable cv;

    Channel(int);
};