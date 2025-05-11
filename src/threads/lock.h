#pragma once

#include <thread>
#include <map>
#include "string"


#include <atomic>

class SimpleMutex {
    std::atomic_flag lock_flag = ATOMIC_FLAG_INIT;

public:
    void lock(); 

    void unlock(); 
};


extern SimpleMutex main_mutex;


extern pthread_mutex_t mutex, clean_scope_mutex, char_pool_mutex, vocab_mutex, random_seed_mutex, aux_mutex, create_thread_mutex;
extern std::map<std::string, pthread_mutex_t *> lockVars;


extern "C" void LockMutex(char *mutex_name);
extern "C" void UnlockMutex(char *mutex_name);
