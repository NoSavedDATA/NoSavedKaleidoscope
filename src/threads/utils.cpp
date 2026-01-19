#include <iostream>
#include <map>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>

#include "../mangler/scope_struct.h"




std::map<std::string, std::map<std::string, void *>> dived_voids;
std::map<std::string, std::map<std::string, int>> dived_ints;
std::map<std::string, std::map<std::string, float>> dived_floats;

extern "C" void dive_void(char *fn_name, void *val, char *var_name) {
    // std::cout << "GOT TO DIVE VOID: " << fn_name << "/" << var_name << ".\n";
    dived_voids[fn_name][var_name] = val;
    // std::cout << "Got " << val << ".\n";
}
extern "C" void dive_int(char *fn_name, int val, char *var_name) {
    dived_ints[fn_name][var_name] = val;
}
extern "C" void dive_float(char *fn_name, float val, char *var_name) {
    dived_floats[fn_name][var_name] = val;
}



extern "C" void *emerge_void(char *fn_name, char *var_name) {

    // std::cout << "----EMERGE VOID: " << fn << ".\n";    
    void *val = dived_voids[fn_name][var_name];
    // std::cout << "got: " << val << ".\n";
    return val;    
}
extern "C" int emerge_int(char *fn_name, char *var_name) {
    return dived_ints[fn_name][var_name];
}
extern "C" float emerge_float(char *fn_name, char *var_name) {
    return dived_floats[fn_name][var_name];    
}





int last_thread_id=1;

extern "C" int tid(Scope_Struct *scope_struct) {
    int thread_id = scope_struct->thread_id-1;
    // long id = syscall(SYS_gettid);
    // std::cout << "tid: " << id << "\n";
    return thread_id;
}




#ifdef _WIN32
  #include <windows.h>
  using pthread_t = HANDLE;
  using pthread_attr_t = void*; // dummy, Windows doesn't use this

extern "C" void pthread_create_aux(pthread_t *thread, pthread_attr_t *attr,
                   void *(*function_ptr) (void *arg), void *arg)
{
    // CreateThread expects LPTHREAD_START_ROUTINE which returns DWORD and takes LPVOID
    // Need to adapt the function pointer signature
    auto wrapper = [](LPVOID arg) -> DWORD {
        auto func_and_arg = static_cast<std::pair<void* (*)(void*), void*>*>(arg);
        void* result = func_and_arg->first(func_and_arg->second);
        delete func_and_arg;
        return 0; // ignore return value
    };

    auto func_and_arg = new std::pair<void* (*)(void*), void*>(function_ptr, arg);

    *thread = CreateThread(
        NULL,
        0,
        wrapper,
        func_and_arg,
        0,
        NULL
    );
}

extern "C" void pthread_join_aux(pthread_t thread)
{
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
}



#else


extern "C" void pthread_create_aux(pthread_t *thread, pthread_attr_t *attr,
                   void *(*function_ptr) (void *arg), void *arg)
{
  pthread_create(thread, attr, function_ptr, arg);
  // std::cout << "Created" << "\n";
}


extern "C" void pthread_join_aux(pthread_t thread)
{
  // std::cout << "Joining " << thread <<  "\n";
  void **value_ptr;
  value_ptr = nullptr;

  pthread_join(thread, value_ptr);
  // std::cout << "Joined: " << thread << "\n";
}

#endif
