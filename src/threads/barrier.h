#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

#pragma once

class Barrier {
public:
    Barrier(int count);
    
    void wait(); 
    
private:
    std::mutex mutex;
    std::condition_variable cv;
    int thread_count;
    int counter;
    int waiting;
};