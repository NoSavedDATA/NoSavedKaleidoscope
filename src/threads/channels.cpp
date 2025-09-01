
#include <iostream>
#include <map>
#include <string>



#include "barrier.h"
#include "channels.h"




Channel::Channel() {

    data_list = new DT_list();
}




extern "C" void *str_channel_message(Scope_Struct *scope_struct, char *str, Channel *ch) {

    std::cout << "--exec: str_channel_message" << ".\n";


    
    str = ch->data_list->unqueue<char*>();
    std::cout << "unpacked: " << str << " from channel " << ch->name << ".\n";

    return str;
}


extern "C" void channel_str_message(Scope_Struct *scope_struct, Channel *ch, char *str) {


    ch->data_list->append(std::any(str), "str");
    

    
}




extern "C" void *channel_Create(Scope_Struct *scope_struct) {


    Channel *ch = new Channel();

    ch->name = "4ch";

    return ch;
}


