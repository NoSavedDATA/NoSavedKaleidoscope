#include <thread>
#include "include.h"

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