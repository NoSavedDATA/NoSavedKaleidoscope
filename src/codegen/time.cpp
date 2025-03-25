#include <random>
#include <iostream>
#include <thread>
#include <chrono>

#include"include.h"





extern "C" void __slee_p_(float id)
{
  std::cout << "\n\nSleep " << id << " begin" << "\n";
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> dis(3, 7); // Generate between 1 and 100
  int random_number = dis(gen);

  //std::this_thread::sleep_for(std::chrono::seconds(random_number));
  std::this_thread::sleep_for(std::chrono::seconds((int)id));

  std::cout << "Sleep " << id << " finish" << "\n";

  //return id;
}


extern "C" float silent_sleep(float id)
{
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<> dis(3, 7); // Generate between 1 and 100
  int random_number = dis(gen);

  //std::this_thread::sleep_for(std::chrono::seconds(random_number));
  std::this_thread::sleep_for(std::chrono::seconds((int)id));

 
  return 0;
}


std::chrono::high_resolution_clock::time_point START_TIME;

extern "C" float start_timer(float id)
{
  START_TIME = std::chrono::high_resolution_clock::now();
 
  return 0;
}

extern "C" float end_timer(float id)
{
  std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsedTime = endTime - START_TIME;

  // Print the elapsed time in seconds
  std::cout << "Elapsed time: " << elapsedTime.count() << " seconds.\n";

  //std::cout << "Length " << rds.size() << "\n";

  return 0;
}