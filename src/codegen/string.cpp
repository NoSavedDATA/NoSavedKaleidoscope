#include<string>
#include<vector>
#include<map>
#include<iostream>

#include"../char_pool/include.h"
#include"include.h"


std::vector<std::string> scopes;

char *RandomString(size_t length) {
  //unsigned int seed = generate_custom_seed();

  const std::string charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  pthread_mutex_lock(&random_seed_mutex);

  //MT19937 mt(generate_custom_seed());
  //LCG rng(generate_custom_seed());

  char *random_string = new char[length+1];

  for (int i = 0; i < length; i++) {

      //int random_index = mt.extract() % charset.length();
      int random_index = rng.next() % charset.length();
      random_string[i] = charset[random_index];
  }

  //random_string[length] = '\0';

  //std::cout << "" << random_string << "\n";

  pthread_mutex_unlock(&random_seed_mutex);

  
  //std::string aux = random_string;
  //if(!in_str(aux,rds))
  //  rds.push_back(aux);
  

  return random_string;
}

extern "C" char * RandomStrOnDemand()
{ 
  return RandomString(14);
}


extern "C" char *GetEmptyChar()
{
  char *empty_char = get_from_char_pool(1,"get empty char");
  empty_char[0] = '\0';
  return empty_char;
}

extern "C" void FreeCharFromFunc(char *_char, char *func) {
  std::cout << "FREEING " << _char << " at function: " << func << "\n";
  delete[] _char;
  std::cout << "freed" << "\n";
}


extern "C" void FreeChar(char *_char) {
  std::cout << "free" << ".\n";
  std::cout << "FREEING " << _char << "\n";

  move_to_char_pool(strlen(_char)+1, _char, "free");
  //delete[] _char;
}






extern "C" char *CopyString(char *in_str)
{

  size_t length = strlen(in_str) + 1;
  char *copied = get_from_char_pool(length, "copy");
  memcpy(copied, in_str, length);

  // std::cout << "copy " << in_str << "\n";

  return copied;
}

extern "C" char * ConcatStr(char *lc, char *rc)
{
  // std::cout << "Concat fn" << ".\n";
  // std::cout << "Concat: " << lc << " -- " << rc << ".\n";
  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  char *result_cstr = get_from_char_pool(length_lc+length_rc, "concat");
  //char* result_cstr = new char[length_lc+length_rc]; 
  
  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);

  //std::cout << "ConcatStr " << result_cstr << "\n";

  return result_cstr;
}

extern "C" char * ConcatStrFreeLeft(char *lc, char *rc)
{
  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  //char* result_cstr = new char[length_lc+length_rc];
  char *result_cstr = get_from_char_pool(length_lc+length_rc, "concat free left");
  
  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);


  move_to_char_pool(length_lc+1, lc, "concat free left");
  //delete[] lc;

  //std::cout << "ConcatStrFreeLeft " << result_cstr << "\n";
  
  return result_cstr;
}

extern "C" char * ConcatStrFreeRight(char *lc, char *rc)
{
  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  //char* result_cstr = new char[length_lc+length_rc];
  char *result_cstr = get_from_char_pool(length_lc+length_rc, "concat free right");
  
  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);

  move_to_char_pool(length_rc, rc, "concat free right");
  //delete[] rc;

  //std::cout << "ConcatStrFreeRight " << result_cstr << "\n";
  
  return result_cstr;
}

extern "C" char * ConcatStrFree(char *lc, char *rc)
{
  size_t length_lc = strlen(lc);
  size_t length_rc = strlen(rc) + 1; // +1 for null terminator
  char* result_cstr = new char[length_lc+length_rc]; 
  
  memcpy(result_cstr, lc, length_lc);
  memcpy(result_cstr + length_lc, rc, length_rc);

  move_to_char_pool(length_lc+1, lc, "concat free");
  move_to_char_pool(length_rc, rc, "concat free");
  //delete[] lc, rc;
  
  //std::cout << "ConcatStrFree " << result_cstr << "\n";

  return result_cstr;
}


extern "C" char * ConcatFloatToStr(char *lc, float r)
{

  //TODO: Change and test the function below
  /*
    char buffer[32]; // 32 bytes should be enough to hold float as a string
    int len = snprintf(buffer, sizeof(buffer), "%.6f", r); // Format float as string with 6 decimal places

    // Calculate the total length for the result
    size_t lc_len = std::strlen(lc);
    size_t total_len = lc_len + len + 1; // +1 for null terminator

    // Allocate the result buffer
    char* result_cstr = new char[total_len];

    // Copy the input string and the formatted float to the result buffer
    std::memcpy(result_cstr, lc, lc_len);
    std::memcpy(result_cstr + lc_len, buffer, len + 1);
  */

  std::string l = lc;
  int _r = r;

  std::string result_str = l + std::to_string(_r);
  char* result_cstr = new char[result_str.length() + 1];
  std::strcpy(result_cstr, result_str.c_str());

  
  return result_cstr;
}

extern "C" char * ConcatNumToStrFree(char *lc, float r)
{
  //std::cout << "\nCONCAT NUM TO STR " << lc << " & " << std::to_string(r) << "\n";

  //TODO: Change and test the function below
  /*
    char buffer[32]; // 32 bytes should be enough to hold float as a string
    int len = snprintf(buffer, sizeof(buffer), "%.6f", r); // Format float as string with 6 decimal places

    // Calculate the total length for the result
    size_t lc_len = std::strlen(lc);
    size_t total_len = lc_len + len + 1; // +1 for null terminator

    // Allocate the result buffer
    char* result_cstr = new char[total_len];

    // Copy the input string and the formatted float to the result buffer
    std::memcpy(result_cstr, lc, lc_len);
    std::memcpy(result_cstr + lc_len, buffer, len + 1);
  */


  std::string l = lc;
  int _r = r;

  std::string result_str = l + std::to_string(_r);
  char* result_cstr = new char[result_str.length() + 1]; // +1 for null terminator
  std::strcpy(result_cstr, result_str.c_str());

  delete[] lc;
  
  return result_cstr;
}