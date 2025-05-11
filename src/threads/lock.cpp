#include <thread>
#include <iostream>
#include "include.h"

#include <atomic>

void SimpleMutex::lock() {
    // while (lock_flag.test_and_set(std::memory_order_acquire)) {
    // }
    int spins = 0;
    while (lock_flag.test_and_set(std::memory_order_acquire)) {
        spins++;
    }
    if (spins > 0) {
      std::cout << "Spin count: " << spins << "\n";
      // std::exit(0);
    }
}
void SimpleMutex::unlock() {
    lock_flag.clear(std::memory_order_release);
}


SimpleMutex main_mutex;

extern "C" void LockMutex(char *mutex_name)
{
  pthread_mutex_t *_mutex = lockVars[mutex_name];
  pthread_mutex_lock(_mutex);
}

extern "C" void UnlockMutex(char *mutex_name)
{
  pthread_mutex_t *_mutex = lockVars[mutex_name];
  pthread_mutex_unlock(_mutex);
}