
#include <cstddef>
#include <cstring>


#include "../pool/include.h"
#include "channels.h"



Channel::Channel() {
}

void Channel::New(int buffer_size) {
    this->buffer_size = buffer_size;
    
    data_list = new DT_list();
}




extern "C" void *channel_Create(Scope_Struct *scope_struct, int buffer_size) {
    Channel *ch = newT<Channel>(scope_struct, "channel");
    ch->New(buffer_size);
    // ch->name = "4ch";
    return ch;
}


// str msg <- ch
extern "C" void *str_channel_message(Scope_Struct *scope_struct, void *ptr, Channel *ch) {    
    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->terminated || ch->data_list->data->size() > 0; } );    
    if(ch->terminated)
        return ptr;

    char *str = ch->data_list->unqueue<char*>();
    ch->cv.notify_all();


    return str;
}

// ch <- msg
extern "C" float channel_str_message(Scope_Struct *scope_struct, Channel *ch, char *str) {
    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->terminated || ch->data_list->data->size() < ch->buffer_size; } );    
    if(ch->terminated)
        return -1;

    size_t length = strlen(str) + 1;
    char *copied = allocate<char>(scope_struct, length, "str");
    memcpy(copied, str, length);

    ch->data_list->append(std::any(copied), "str");

    ch->cv.notify_all();
    
    return 0;
}


extern "C" void *void_channel_message(Scope_Struct *scope_struct, void *ptr, Channel *ch) {    
    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->terminated || ch->data_list->data->size() > 0; } );    
    if(ch->terminated)
        return ptr;

    void *ret = ch->data_list->unqueue<void*>();
    ch->cv.notify_all();

    return ret;
}

extern "C" float channel_void_message(Scope_Struct *scope_struct, Channel *ch, void *ptr) {
    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->terminated || ch->data_list->data->size() < ch->buffer_size; } );    
    if(ch->terminated)
        return -1;

    ch->data_list->append(std::any(ptr), "any");
    ch->cv.notify_all();
    
    return 0;
}

extern "C" char * str_channel_Idx(Scope_Struct *scope_struct, Channel *ch, int idx) {
    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->terminated || ch->data_list->data->size() > 0; } );    
    if(ch->terminated)
        return nullptr;

    char *str = ch->data_list->unqueue<char*>();

    ch->cv.notify_all();

    return str;
}


extern "C" void str_channel_terminate(Scope_Struct *scope_struct, Channel *ch) {
    {
        std::unique_lock<std::mutex> lock(ch->mtx);
        ch->terminated = true;
    }
    ch->cv.notify_all();  // wake up all waiting senders/receivers
}


extern "C" int str_channel_alive(Scope_Struct *scope_struct, Channel *ch) {
    return !ch->terminated;
}










extern "C" float float_channel_message(Scope_Struct *scope_struct, void *ptr, Channel *ch) {    
    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->terminated || ch->data_list->data->size() > 0; } );
    if(ch->terminated)
        return -1;

    float x = ch->data_list->unqueue<float>();

    ch->cv.notify_all();

    return x;
}

extern "C" float channel_float_message(Scope_Struct *scope_struct, Channel *ch, float x) {
    std::unique_lock<std::mutex> lock(ch->mtx);
    
    ch->cv.wait(lock, [&]{ return ch->terminated || ch->data_list->data->size() < ch->buffer_size; } );
    if(ch->terminated)
        return -1;

    ch->data_list->append(std::any(x), "float");

    ch->cv.notify_all();
    
    return 0;
}


extern "C" float float_channel_Idx(Scope_Struct *scope_struct, Channel *ch, int idx) {
    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->terminated || ch->data_list->data->size() > 0; } );    
    if(ch->terminated)
        return -1;

    float ret = ch->data_list->unqueue<float>();

    ch->cv.notify_all();

    return ret;
}

extern "C" float float_channel_sum(Scope_Struct *scope_struct, Channel *ch) {
    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->terminated || ch->data_list->data->size() <= ch->buffer_size; } );
    if(ch->terminated)
        return -1;


    float sum=0;
    for(int i=0; i<ch->buffer_size; ++i)
        sum += ch->data_list->unqueue<int>();

    ch->cv.notify_all();
    
    return sum;
}

extern "C" float float_channel_mean(Scope_Struct *scope_struct, Channel *ch) {
    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->terminated || ch->data_list->data->size() <= ch->buffer_size; } );
    if(ch->terminated)
        return -1;

    float sum=0;
    for(int i=0; i<ch->buffer_size; ++i)
        sum += ch->data_list->unqueue<int>();

    float mean = sum / (float)ch->buffer_size;

    ch->cv.notify_all();
    
    return mean;
}

extern "C" float float_channel_terminate(Scope_Struct *scope_struct, Channel *ch) {
    {
        std::unique_lock<std::mutex> lock(ch->mtx);
        ch->terminated = true;
    }
    ch->cv.notify_all();  // wake up all waiting senders/receivers
    return 0;
}

extern "C" int float_channel_alive(Scope_Struct *scope_struct, Channel *ch) {
    return !ch->terminated;
}






// int x <- ch
extern "C" int int_channel_message(Scope_Struct *scope_struct, void *ptr, Channel *ch) {    
    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->terminated || ch->data_list->data->size() > 0; } );
    if(ch->terminated)
        return -1;

    int x = ch->data_list->unqueue<int>();

    ch->cv.notify_all();

    return x;
}

// ch <- msg
extern "C" float channel_int_message(Scope_Struct *scope_struct, Channel *ch, int x) {
    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->terminated || ch->data_list->data->size() < ch->buffer_size; } );
    if(ch->terminated)
        return -1;

    ch->data_list->append(std::any(x), "int");

    ch->cv.notify_all();
    
    return 0;
}

extern "C" int int_channel_Idx(Scope_Struct *scope_struct, Channel *ch, int idx) {
    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->terminated || ch->data_list->data->size() > 0; } );    
    if(ch->terminated)
        return -1;

    int res = ch->data_list->unqueue<int>();

    ch->cv.notify_all();

    return res;
}

extern "C" int int_channel_sum(Scope_Struct *scope_struct, Channel *ch) {
    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->terminated || ch->data_list->data->size() <= ch->buffer_size; } );
    if(ch->terminated)
        return -1;

    int sum=0;
    for(int i=0; i<ch->buffer_size; ++i)
        sum += ch->data_list->unqueue<int>();

    ch->cv.notify_all();
    
    return sum;
}

extern "C" float int_channel_mean(Scope_Struct *scope_struct, Channel *ch) {
    std::unique_lock<std::mutex> lock(ch->mtx);

    ch->cv.wait(lock, [&]{ return ch->terminated || ch->data_list->data->size() <= ch->buffer_size; } );
    if(ch->terminated)
        return -1;

    int sum=0;
    for(int i=0; i<ch->buffer_size; ++i)
        sum += ch->data_list->unqueue<int>();

    float mean = (float)sum / (float)ch->buffer_size;

    ch->cv.notify_all();
    
    return mean;
}


extern "C" float int_channel_terminate(Scope_Struct *scope_struct, Channel *ch) {
    {
        std::unique_lock<std::mutex> lock(ch->mtx);
        ch->terminated = true;
    }
    ch->cv.notify_all();  // wake up all waiting senders/receivers
    return 0;
}

extern "C" bool int_channel_alive(Scope_Struct *scope_struct, Channel *ch) {
    return !ch->terminated;
}


void channel_Clean_Up(void *ptr) {
}
