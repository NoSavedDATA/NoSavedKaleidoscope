#include "util.h"





void Grid::NewGrid(int gx, int gy, int bx, int by)
{
  this->g.x = gx;
  this->g.y = gy;
  this->g.z = 1;

  this->b.x = bx;
  this->b.y = by;
  this->b.z = 1;

  smem = (bx+by)*32*sizeof(float);
}

void Grid::SetWarpSize(int wx, int wy)
{
  wx_per_bx = b.x / (wx*16);
  wy_per_by = b.y / (wy*16);

  this->w.x = wx*32;
  this->w.y = wy;
  this->w.z = 1;
}

Grid CalculateBlockingSize(int M, int N)
{

  int bx = 256;
  int by = 128;


  while(bx>M && bx>64)
    bx = bx/2;

  while(by>N && by>64)
    by = by/2;

  int gx = std::floor((M+bx-1)/(float)bx);
  int gy = std::floor((N+by-1)/(float)by);

  Grid grid;

  // std::cout << gx << ", " << gy << ", " << bx << ", " << by << "\n";
  grid.NewGrid(gx, gy, bx, by);


  int wx = fminf(fmaxf(M/16,1),4);
  int wy = fminf(fmaxf(N/16,1),4);

  wx = 4;
  wy = 2;

  grid.SetWarpSize(wx, wy);

  return grid;
}











  



void Grid2::NewGrid(int gx, int gy, int bx, int by)
{
  this->g.x = gx;
  this->g.y = gy;
  this->g.z = 1;

  this->b.x = bx;
  this->b.y = by;
  this->b.z = 1;

  smem = (bx+by)*64*sizeof(float); //times twice wk
}

void Grid2::SetWarpSize(int warps, int wx, int wy)
{
  bx_per_w = b.x / warps;
  by_per_w = b.y / warps;

  this->w.x = warps*32;


  this->wx = wx;
  this->wy = wy;


  bx_per_wx = this->b.x/wx; // each bx work is splitted accross wx

  wx_per_wmma_m = wx / 16;
  wy_per_wmma_n = wy / 16;

}


Grid2 CalculateBlockingSize2(int M, int N)
{

  int bx = 128;
  int by = 64;


  // while(bx>M && bx>64)
  //   bx = bx/2;

  // while(by>N && by>64)
  //   by = by/2;

  int gx = std::floor((M+bx-1)/(float)bx); // Each gx handles 128 rows of x (bx)
  int gy = std::floor((N+by-1)/(float)by);

  Grid2 grid;

  // std::cout << gx << ", " << gy << ", " << bx << ", " << by << "\n";
  grid.NewGrid(gx, gy, bx, by);


  // int wx = fminf(fmaxf(M/16,1),4);
  // int wy = fminf(fmaxf(N/16,1),4);

  int warps = 8;
  int wx = 32; // each wx handles 32 rows
  int wy = 32;

  grid.SetWarpSize(warps, wx, wy);

  return grid;
}



