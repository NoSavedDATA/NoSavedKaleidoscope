
#include <iostream>
#include <map>
#include <string>
#include <vector>


// #include "../compiler_frontend/tokenizer.h"
#include "../compiler_frontend/include.h"
#include "../common/extension_functions.h"
#include "data_tree.h"


std::map<std::string, Data_Tree> functions_return_data_type;


Data_Tree::Data_Tree(std::string Type, std::vector<Data_Tree> Nested_Data) : Type(Type), Nested_Data(std::move(Nested_Data)) {}
Data_Tree::Data_Tree(std::string Type) : Type(Type) {}


bool CheckChannel(Data_Tree *L_ptr, Data_Tree R) {
    Data_Tree L = *L_ptr;

    if(!(L.Type=="channel"||R.Type=="channel"))
        return false;
    
    if(L.Type=="channel")
        L = L.Nested_Data[0];
    if(R.Type=="channel")
        R = R.Nested_Data[0];
    
    return L.Compare(R)==0;
}

bool CompareListTuple(Data_Tree *L, Data_Tree R) {

    if(L->Type!="list"||R.Type!="tuple")
        return true;


    std::string list_type = L->Nested_Data[0].Type;


    for (auto data_tree : R.Nested_Data)
    {
        if(list_type!=data_tree.Type)
            return false;
    }


    return true;
}

int Data_Tree::Compare(Data_Tree other_tree) {
    
    int comparisons = 0;


    if(!in_str(Type, {primary_data_tokens}) && other_tree.Type=="nullptr")
        return comparisons;
 

 
    if(Type!=other_tree.Type&&!CompareListTuple(this, other_tree)&&!CheckIsEquivalent(Type, other_tree.Type))
        comparisons++;

    if (Type=="list"&&other_tree.Type=="tuple"||CheckChannel(this, other_tree))
        return comparisons;

    if(Nested_Data.size()!=other_tree.Nested_Data.size()){
        std::cout << "nested data has different size: " << Nested_Data.size() << "/" << other_tree.Nested_Data.size() << ".\n";
        comparisons++;
    }
    else {
        for(int i=0; i<Nested_Data.size(); ++i)
            comparisons += Nested_Data[i].Compare(other_tree.Nested_Data[i]);
    }

    return comparisons;
}

void Data_Tree::Print() {
    std::cout << "" << Type;
    if (Nested_Data.size()>0)
    {
        std::cout << "<";
        Nested_Data[0].Print();
        for(int i=1; i<Nested_Data.size(); ++i)
        {
            std::cout << ",";
            Nested_Data[i].Print();
        }
        std::cout << ">";
    }
}








std::string UnmangleVec(Data_Tree dt) {
    if (dt.Type=="vec")
        return  dt.Nested_Data[0].Type + "_vec";
    return dt.Type;
}