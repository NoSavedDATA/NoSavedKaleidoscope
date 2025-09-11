#include <cstdlib>
#include <cstring>

extern "C" char *bool_to_str(bool x) {
    const char *word = x ? "true" : "false";
    size_t len = std::strlen(word);

    char *result = (char *)std::malloc(len + 1);
    if (!result) return nullptr;

    std::memcpy(result, word, len + 1); // copy including '\0'
    return result;
}

