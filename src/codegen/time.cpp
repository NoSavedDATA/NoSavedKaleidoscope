#include <random>
#include <iostream>
#include <thread>
#include <chrono>
#include <map>

#include "../mangler/scope_struct.h"


#ifdef _WIN32
#include <windows.h>

unsigned int get_millisecond_time() {
    // Returns milliseconds since system start (wraps around after ~49 days)
    return static_cast<unsigned int>(GetTickCount64() & 0xFFFFFFFF);
}
#else
#include <ctime>

unsigned int get_millisecond_time() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<unsigned int>(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
#endif


extern "C" void __slee_p_(Scope_Struct *scope_struct, int duration)
{
  // std::cout << "\n\nSleep " << duration << " begin" << "\n";
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> dis(3, 7); // Generate between 1 and 100
  int random_number = dis(gen);

  //std::this_thread::sleep_for(std::chrono::seconds(random_number));
  std::this_thread::sleep_for(std::chrono::seconds(duration));

  // std::cout << "Sleep " << duration << " finish" << "\n";

  //return id;
}


extern "C" void random_sleep(Scope_Struct *scope_struct, int start, int finish)
{
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> dis(start, finish); // Generate between 1 and 100
  int random_number = dis(gen);

  //std::this_thread::sleep_for(std::chrono::seconds(random_number));
  std::this_thread::sleep_for(std::chrono::seconds(random_number));


  //return id;
}


extern "C" float silent_sleep(Scope_Struct *scope_struct, int duration)
{
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> dis(3, 7); // Generate between 1 and 100
  int random_number = dis(gen);

  //std::this_thread::sleep_for(std::chrono::seconds(random_number));
  std::this_thread::sleep_for(std::chrono::seconds(duration));

 
  return 0;
}


std::chrono::high_resolution_clock::time_point START_TIME;

extern "C" float start_timer(Scope_Struct *scope_struct)
{
  START_TIME = std::chrono::high_resolution_clock::now();
 
  return 0;
}

extern "C" float end_timer(Scope_Struct *scope_struct)
{
  std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsedTime = endTime - START_TIME;

  // Print the elapsed time in seconds
  std::cout << "Elapsed time: " << elapsedTime.count() << " seconds.\n";

  //std::cout << "Length " << rds.size() << "\n";

  return 0;
}