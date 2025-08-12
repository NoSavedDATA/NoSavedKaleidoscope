#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>




struct SpinLock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;

    void lock();
    void unlock();

};



extern std::unordered_map<std::string, SpinLock *> lockVars;


extern "C" void LockMutex(char *mutex_name);
extern "C" void UnlockMutex(char *mutex_name);
