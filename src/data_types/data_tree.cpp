
#include <iostream>
#include <map>
#include <string>
#include <vector>


// #include "../compiler_frontend/tokenizer.h"
#include "../compiler_frontend/logging_v.h"
#include "../compiler_frontend/include.h"
#include "../common/extension_functions.h"
#include "data_tree.h"


std::map<std::string, Data_Tree> functions_return_data_type;


Data_Tree::Data_Tree(std::string Type, std::vector<Data_Tree> Nested_Data) : Type(Type), Nested_Data(std::move(Nested_Data)) {}
Data_Tree::Data_Tree(std::string Type) : Type(Type) {}

bool CompareListUnkList(Data_Tree *L, Data_Tree R) {
    if((L->Type=="list"&&R.Type=="unknown_list")||L->Type=="unknown_list"&&R.Type=="list")
        return true;
    return false;
}


bool CompareListRecursive(Data_Tree L, Data_Tree R) {
    
    if (L.Nested_Data.size()==0) //todo: check this condition
        return true;

    std::string list_type = L.Nested_Data[0].Type;

    if (in_str(list_type,{"list", "tuple", "dict", "array"})) {
        if (!in_str(R.Type,{"list","tuple","dict","array"}))
            return false;
        return CompareListRecursive(L.Nested_Data[0], R.Nested_Data[0]);
    }

    if(R.Type=="list")
        return list_type==R.Nested_Data[0].Type;
    
    if(R.Type=="tuple") {
        for (auto data_tree : R.Nested_Data)
        {
            if(list_type!=data_tree.Type)
                return false;
        }    
    }
    
    
    return true;
}

bool Data_Tree::CompareMap(Data_Tree &R) {
    return CheckIsEquivalent(Type, R.Nested_Data[1].Type);
}

bool CompareListTuple(Data_Tree *L, Data_Tree R) {

    if(L->Type!="list"||R.Type!="tuple")
        return true;

    CompareListRecursive(*L, R);
    return true;
}



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



int Data_Tree::Compare(Data_Tree other_tree) {    
    int comparisons = 0;

    if(Type=="void")
        return 0;

    if(!in_str(Type, primary_data_tokens) && other_tree.Type=="nullptr")
        return 0;

    if(Type=="any"||other_tree.Type=="any")
        return 0;

    if(Type=="array"&&other_tree.Type=="array")
        return 0;

    if(Type!="map"&&other_tree.Type=="map"&&CompareMap(other_tree))
        return 0;

    if(Type=="map"&&other_tree.Type!="map"&&Nested_Data[1].Type==other_tree.Type)
        return 0;

 
    if((Nested_Data.size()==0&&other_tree.Nested_Data.size()==0) && !CheckIsEquivalent(Type, other_tree.Type))
        return comparisons+1;
     
    if(!CompareListTuple(this, other_tree))
        return comparisons+1;

    if ((Type=="list"||Type=="array")&&other_tree.Type=="tuple"||CheckChannel(this, other_tree))
        return comparisons;

    if(Nested_Data.size()!=other_tree.Nested_Data.size()){
        if ((Type=="list"&&other_tree.Type=="list")&&(Nested_Data.size()>0&&other_tree.Nested_Data.size()==0))
            return comparisons;

        LogErrorC(-1, "Nested data has different size: " + std::to_string(Nested_Data.size()) + \
				      "/" + std::to_string(other_tree.Nested_Data.size()) + ".\n");
        comparisons++;
    } else {
        for(int i=0; i<Nested_Data.size(); ++i)
            comparisons += Nested_Data[i].Compare(other_tree.Nested_Data[i]);
    }

    return comparisons;
}

void Data_Tree::Print() {
    std::string str = toString();
    std::cout << str << "\n";
}


std::string Data_Tree::toString() {
    std::string str = Type; 
    if (Nested_Data.size()>0)
    {
        str += "<";
        Data_Tree dt = Nested_Data[0];
        str += dt.toString();
        for (int i=1; i<Nested_Data.size(); ++i) {
            dt = Nested_Data[i];
            str += ",";
            str += dt.toString();
        }
        str += ">";

    }
    return str;
}






std::string UnmangleVec(Data_Tree dt) {
    if (dt.Type=="vec")
        return  dt.Nested_Data[0].Type + "_vec";
    if (dt.Type=="channel")
        return  dt.Nested_Data[0].Type + "_channel";
    return dt.Type;
}
