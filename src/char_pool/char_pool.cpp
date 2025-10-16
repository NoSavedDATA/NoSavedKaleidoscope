
#include "include.h"

void move_to_char_pool(size_t length, char *char_ptr, std::string from)
{
  free(char_ptr);
  return;
  
  if (length==0)
  {
    // std::cout << "RETURNING FROM CHAR POOL CAUSE SIZE 0" << ".\n";
    return;
  }
  //std::cout << "\nmove_to_char_pool from: " << from << "\n";
  

  std::vector<char *> chars_in_pool = CharPool[length];
  if (!in_char_ptr_vec(char_ptr, chars_in_pool))
  {
    //if(!(chars_in_pool.size()<30&&length==1))
    if(chars_in_pool.size()<270)
    {
      // std::cout << "MOVE TO CHAR OF SIZE " << length << " TO THE POOL.\n";
      CharPool[length].push_back(char_ptr);
    }
    else
    {
      // std::cout << "FREEING CHAR WITH length: " << length << " from: " << from <<  "\n";
      free(char_ptr);
    }
  } 
}

char *get_from_char_pool(size_t length, std::string from)
{
  if (length==0)
    return nullptr;
  char *char_ptr;


  char_ptr = (char*)malloc(length);
  return char_ptr;

  // std::cout << "GETTING CHAR OF SIZE " << length << " TO POOL.\n";

  
  if(CharPool.count(length)>0)
  {
    std::vector<char *> chars_in_pool = CharPool[length];
    if (chars_in_pool.size()>0)
    {
      //std::cout << "GETTING FROM CHAR POOL: " << length << "\n";
      char_ptr = chars_in_pool.back();
      CharPool[length].pop_back();
      return char_ptr;
    }
  }

  // std::cout << "\nMalloc new CHAR from " << from << " of size: " << length << "\n";

  char_ptr = (char*)malloc(length);
  return char_ptr;
}