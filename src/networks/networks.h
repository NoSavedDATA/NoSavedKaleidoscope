#pragma once


#include "networks.h"


extern "C" float network_ema(int thread_id, char *scope, char *_ema_network, char *_network, float factor);
