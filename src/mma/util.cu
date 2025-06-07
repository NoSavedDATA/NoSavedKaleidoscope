#include "util.h"




Wmma_Grid::Wmma_Grid(int gx, int gy, int warps, int bx, int by, int wx, int wy, int wmma_m, int wmma_n)
                  : bx(bx), by(by), wx(wx), wy(wy), warps(warps)
{
  this->g.x = gx;
  this->g.y = gy;

  // smem = (bx+by)*64*sizeof(float); //times twice wk
  smem = (128+64)*64*sizeof(float); //times twice wk


  bx_per_w = bx / warps; // each bx work is splitted accross warps
  by_per_w = by / warps;

  this->w.x = warps*32;

  bx_per_wx = bx/wx;

  wx_per_wmma_m = wx / wmma_m;
  wy_per_wmma_n = wy / wmma_n;
}



Wmma_Grid CalculateBlockingSize(int M, int N,
                                int warps,
                                int block_size_x, int block_size_y,
                                int wx, int wy,
                                int wmma_m, int wmma_n)
{  
  // while(bblock_size_x>M && block_size_x>64)
  //   block_size_x = block_size_x/2;
  // while(block_size_y>N && block_size_y>64)
  //   block_size_y = block_size_y/2;

  int gx = std::floor((M+block_size_x-1)/(float)block_size_x); // Each gx handles 128 rows of x (bx)
  int gy = std::floor((N+block_size_y-1)/(float)block_size_y);

  // int wx = fminf(fmaxf(M/16,1),4);
  // int wy = fminf(fmaxf(N/16,1),4);

  Wmma_Grid grid(gx, gy,
                 warps,
                 block_size_x, block_size_y,
                 wx, wy,
                 wmma_m, wmma_n);

  return grid;
}



