#include <filesystem>

extern "C" float dir_exists(char *path)
{
    if (std::filesystem::exists(path) && std::filesystem::is_directory(path))
    return 1;

  return 0;
}

extern "C" float path_exists(char *path)
{
  if (std::filesystem::exists(path))
    return 1;

  return 0;
}