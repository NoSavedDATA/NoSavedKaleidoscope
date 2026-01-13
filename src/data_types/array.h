#pragma once
#include<string>

class DT_array {
    public:
		int virtual_size, size, elem_size;
		void *data;
        std::string type;

    DT_array();
    void New(int, int, std::string);
    void New(int, std::string);
};


void array_Clean_Up(void *data_ptr);

