#include <cstdlib>


extern "C" float _exit() {
    std::exit(0);
    return 0;
}