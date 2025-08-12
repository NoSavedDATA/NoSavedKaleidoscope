#include <thread>
#include <iostream>
#include "include.h"

#include <atomic>



void SpinLock::lock() {
    while (flag.test_and_set(std::memory_order_acquire)) {
        // Optional: yield or sleep a bit to reduce CPU usage
        std::this_thread::yield();
    }
}
void SpinLock::unlock() {
    flag.clear(std::memory_order_release);
}


std::mutex map_mutex;
std::unordered_map<std::string, SpinLock *> lockVars;

extern "C" void LockMutex(char* mutex_name)
{
    std::string key(mutex_name);
    {
        std::lock_guard<std::mutex> guard(map_mutex);
        auto it = lockVars.find(key);
        if (it == lockVars.end())
        {
            // allocate SpinLock dynamically
            SpinLock* new_lock = new SpinLock();
            lockVars.emplace(std::move(key), new_lock);
            it = lockVars.find(key);
        }
        // else it already exists
        it->second->lock();
    }
}

extern "C" void UnlockMutex(char* mutex_name)
{
    std::string key(mutex_name);
    std::lock_guard<std::mutex> guard(map_mutex);
    auto it = lockVars.find(key);
    if (it != lockVars.end())
        it->second->unlock();
}