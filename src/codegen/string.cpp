#include<string>
#include<vector>
#include<map>

#include"include.h"



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