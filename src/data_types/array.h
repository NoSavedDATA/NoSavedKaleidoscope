#pragma once

class DT_array {
    public:
		int virtual_size, size, elem_size;
		void *data;

    DT_array();
    void New(int, int);
};
