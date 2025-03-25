#pragma once



extern pthread_mutex_t mutex, clean_scope_mutex, char_pool_mutex, vocab_mutex, random_seed_mutex, aux_mutex;
extern std::map<std::string, pthread_mutex_t *> lockVars;