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

    Channel();
    void New(int);
};

void channel_Clean_Up(void *ptr);


void channel_handle_pool(Scope_Struct *scope_struct, void *ptr, char *data_name);
