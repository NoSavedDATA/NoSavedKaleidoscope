#pragma once

#include"include.h"



char *RandomString(size_t length); 

extern "C" char * RandomStrOnDemand();


extern "C" char *GetEmptyChar();


extern "C" void FreeCharFromFunc(char *_char, char *func); 


extern "C" void FreeChar(char *_char); 


extern "C" char *CopyString(char *in_str);


extern "C" char * ConcatStr(char *lc, char *rc);

extern "C" char * ConcatStrFreeLeft(char *lc, char *rc);

extern "C" char * ConcatStrFreeRight(char *lc, char *rc);

extern "C" char * ConcatStrFree(char *lc, char *rc);

extern "C" char * ConcatFloatToStr(char *lc, float r);

extern "C" char * ConcatNumToStrFree(char *lc, float r);

