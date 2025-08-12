#pragma once




extern int last_thread_id;







#ifdef _WIN32
  #include <windows.h>
  using pthread_t = HANDLE;
  using pthread_attr_t = void*; // dummy, Windows doesn't use this

extern "C" void pthread_create_aux(pthread_t *thread, pthread_attr_t *attr,
                   void *(*function_ptr) (void *arg), void *arg);

extern "C" void pthread_join_aux(pthread_t thread);


#else

extern "C" void pthread_create_aux(pthread_t *thread, pthread_attr_t *attr,
                   void *(*function_ptr) (void *arg), void *arg);

extern "C" void pthread_join_aux(pthread_t thread);

#endif