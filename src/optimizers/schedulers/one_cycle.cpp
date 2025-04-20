
#include <cmath>

#include "../../mangler/scope_struct.h"

extern "C" float OneCycleLR(Scope_Struct *scope_struct, float base_lr, float step, float max_steps)
{
  // Possibly wrong.
  float pct_start, final_div_factor, max_momentum, min_momentum, cycle_length, down_phase_steps, min_lr;
  pct_start=0.3;
  final_div_factor=1000;
  max_momentum=0.95;
  min_momentum=0.85;

  cycle_length = int(max_steps*pct_start);
  down_phase_steps = max_steps - cycle_length;

  min_lr = base_lr/final_div_factor;

  
  if(step<cycle_length)
    return base_lr * step/cycle_length;
  if(step<max_steps)
    return base_lr * (1 + std::cos(M_PI * (step - cycle_length) / (max_steps - cycle_length))) / 2;
  return min_lr;
}