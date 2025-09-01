
#include <cstddef>
#include <cstring>


#include "channels.h"



Channel::Channel(int buffer_size) : buffer_size(buffer_size) {

    data_list = new DT_list();
}




extern "C" void *str_channel_message(Scope_Struct *scope_struct, void *ptr, Channel *ch) {    
    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->data_list->data->size() > 0; } );    

    char *str = ch->data_list->unqueue<char*>();

    ch->cv.notify_one();

    return str;
}

extern "C" float channel_str_message(Scope_Struct *scope_struct, Channel *ch, char *str) {
    std::unique_lock<std::mutex> lock(ch->mtx);
    ch->cv.wait(lock, [&]{ return ch->data_list->data->size() < ch->buffer_size; } );    

    size_t length = strlen(str) + 1;
    char *copied = (char*)malloc(length);
    memcpy(copied, str, length);

    ch->data_list->append(std::any(copied), "str");

    ch->cv.notify_one();
    
    return 0;
}


extern "C" int int_channel_message(Scope_Struct *scope_struct, void *ptr, Channel *ch) {    
    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->data_list->data->size() > 0; } );

    int x = ch->data_list->unqueue<int>();

    ch->cv.notify_one();

    return x;
}

extern "C" float channel_int_message(Scope_Struct *scope_struct, Channel *ch, int x) {
    std::unique_lock<std::mutex> lock(ch->mtx);
    ch->cv.wait(lock, [&]{ return ch->data_list->data->size() < ch->buffer_size; } );

    ch->data_list->append(std::any(x), "int");

    ch->cv.notify_one();
    
    return 0;
}



extern "C" float float_channel_message(Scope_Struct *scope_struct, void *ptr, Channel *ch) {    

    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->data_list->data->size() > 0; } );

    float x = ch->data_list->unqueue<float>();

    ch->cv.notify_one();

    return x;
}

extern "C" float channel_float_message(Scope_Struct *scope_struct, Channel *ch, float x) {
    std::unique_lock<std::mutex> lock(ch->mtx);
    ch->cv.wait(lock, [&]{ return ch->data_list->data->size() < ch->buffer_size; } );

    ch->data_list->append(std::any(x), "float");

    ch->cv.notify_one();
    
    return 0;
}




extern "C" void *channel_Create(Scope_Struct *scope_struct, int buffer_size) {
    Channel *ch = new Channel(buffer_size);
    ch->name = "4ch";
    return ch;
}


