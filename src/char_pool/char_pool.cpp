
#include "include.h"

void move_to_char_pool(size_t length, char *char_ptr, std::string from)
{
  delete[] char_ptr;
  return;
  if (length==0)
    return;
  //std::cout << "\nmove_to_char_pool from: " << from << "\n";
  

  pthread_mutex_lock(&char_pool_mutex);
  std::vector<char *> chars_in_pool = CharPool[length];
  if (!in_char_ptr_vec(char_ptr, chars_in_pool))
  {
    //if(!(chars_in_pool.size()<30&&length==1))
    if(chars_in_pool.size()<270)
      CharPool[length].push_back(char_ptr);
    else
    {
      std::cout << "FREEING CHAR WITH length: " << length << " from: " << from <<  "\n";
      delete[] char_ptr;
    }
  } 
  pthread_mutex_unlock(&char_pool_mutex);
}

char *get_from_char_pool(size_t length, std::string from)
{
  if (length==0)
    return nullptr;


  char *char_ptr;
  char_ptr = (char*)malloc(length);
  return char_ptr;

  
  pthread_mutex_lock(&char_pool_mutex);
  if(CharPool.count(length)>0)
  {
    std::vector<char *> chars_in_pool = CharPool[length];
    if (chars_in_pool.size()>0)
    {
      //std::cout << "GETTING FROM CHAR POOL: " << length << "\n";
      char_ptr = chars_in_pool.back();
      CharPool[length].pop_back();
      pthread_mutex_unlock(&char_pool_mutex);
      return char_ptr;
    }
  }
  pthread_mutex_unlock(&char_pool_mutex);

  //std::cout << "\nMalloc new CHAR from " << from << " of size: " << length << "\n";

  char_ptr = (char*)malloc(length);
  return char_ptr;
}