#pragma once

#include <thread>
#include <map>
#include "string"



extern pthread_mutex_t mutex, clean_scope_mutex, char_pool_mutex, vocab_mutex, random_seed_mutex, aux_mutex;
extern std::map<std::string, pthread_mutex_t *> lockVars;


extern "C" void LockMutex(char *mutex_name);
extern "C" void UnlockMutex(char *mutex_name);
