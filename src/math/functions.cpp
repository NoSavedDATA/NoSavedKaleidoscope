#include <cmath>


extern "C" float min(float l, float r)
{
  if (l<r)
    return l;
  return r;
}
extern "C" float max(float l, float r)
{
  if (l>r)
    return l;
  return r;
}
extern "C" float logE2f(float v) {
  return log2f(v);
}
extern "C" float roundE(float v) {
  return round(v);
}
extern "C" float floorE(float v) {
  return floor(v);
}
extern "C" float logical_not(float v)
{
  if (v==0.0f)
    return 1;
  return 0;
}