#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>


struct Data_Tree {
    std::vector<Data_Tree> Nested_Data;
    std::string Type="";
    bool empty=true;
    
    Data_Tree() = default;
    Data_Tree(std::string, std::vector<Data_Tree> nested_data);
    Data_Tree(std::string);

    int Compare(Data_Tree);
    void Print();
};




extern std::map<std::string, Data_Tree> functions_return_data_type;


std::string UnmangleVec(Data_Tree dt);