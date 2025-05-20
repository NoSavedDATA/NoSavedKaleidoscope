#pragma once




extern int last_thread_id;



//int pthread_create(pthread_t *thread, pthread_attr_t *attr,
//                   void *(*start_routine) (void *arg), void *arg);

extern "C" void pthread_create_aux(pthread_t *thread, pthread_attr_t *attr,
                   void *(*function_ptr) (void *arg), void *arg);



extern "C" void pthread_join_aux(pthread_t thread);
