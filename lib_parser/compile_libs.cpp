#include <map>
#include <string>
#include <vector>

#include "include.h"

std::string Get_Base_Lib() {
    std::string fpath = "src/libs_llvm/libs.cpp";


    std::ifstream libs_file(fpath);

    std::string file_string = "";

    if (libs_file.is_open())
    {
        std::string line;
        std::vector<std::string> lines;

        int i = 0;
        while(std::getline(libs_file, line))
        {
            file_string = file_string + line;
            file_string = file_string + '\n';
            i+=1;
            if (i>17)
                break;
        }
            
            // lines.push_back(line);

        libs_file.close();

        // for (const std::string& fileLine : lines)
        //     std::cout << fileLine << std::endl;
    }
    return file_string;
}



int main() {

    Parse_Libs();

    std::string base_lib = Get_Base_Lib();
    

    return 0;
}