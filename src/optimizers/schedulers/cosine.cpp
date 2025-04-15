
#include <cmath>


extern "C" float CosineLR(float base_lr, float min_lr, float step, float max_steps)
{
  //float min_lr = base_lr*0.05;

  if(step<max_steps)
    return min_lr + (base_lr-min_lr) * (1 + std::cos(M_PI * (step/max_steps))) / 2;
  return min_lr;
}