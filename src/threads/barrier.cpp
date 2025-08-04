#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

#include "barrier.h"


Barrier::Barrier(int count) 
    : thread_count(count), counter(0), waiting(0) {}
    
void Barrier::wait() {
    std::unique_lock<std::mutex> lock(mutex);
    
    ++counter;
    ++waiting;
    
    if (counter == thread_count) {
        // Last thread has arrived
        counter = 0;
        cv.notify_all();
    } else {
        cv.wait(lock, [this] { return counter == 0; });
    }
    
    --waiting;
}



extern "C" void *get_barrier(int threads_count) {

    Barrier *barrier = new Barrier(threads_count);

    return barrier;
}
