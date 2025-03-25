#pragma once

#include <vector>

extern "C" float PrintStrVec(std::vector<char*> vec);


extern "C" float LenStrVec(std::vector<char*> vec);




extern "C" void * ShuffleStrVec(std::vector<char*> vec);



//deprecated
extern "C" char * shuffle_str(char *string_list);
