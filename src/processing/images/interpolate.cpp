#include <iostream>
#include <bitset>
#include <climits>
#include <cstring>
#include <iostream>
#include <cmath>

#include <algorithm>
#include <cstdint>
#include <iomanip>

#include "interpolate.h"




unsigned char *interpolate_img(unsigned char *src, int height, int width, int dst_height, int dst_width)
{
    unsigned char *new_img = new unsigned char[dst_height*dst_width*3];


    // Calculate scale factors for width and height
    float xScale = static_cast<float>(width) / dst_width;
    float yScale = static_cast<float>(height) / dst_height;



    for (int y = 0; y < dst_height; ++y) {
        for (int x = 0; x < dst_width; ++x) {
            // Find the corresponding position in the input image
            float srcX = x * xScale;
            float srcY = y * yScale;
            int x1 = static_cast<int>(srcX);
            int y1 = static_cast<int>(srcY);
            int x2 = std::min(x1 + 1, width - 1);
            int y2 = std::min(y1 + 1, height - 1);

            float alphaX = srcX - x1;
            float alphaY = srcY - y1;

            // Get the pixel values from the input image
            auto getPixel = [&](int x, int y, int c) {
                int index = (y * width + x) * 3 + c;
                return static_cast<float>(src[index]);
            };

            // Perform bilinear interpolation
            for (int c = 0; c < 3; ++c) {
                float value1 = (1 - alphaX) * getPixel(x1, y1, c) + alphaX * getPixel(x2, y1, c);
                float value2 = (1 - alphaX) * getPixel(x1, y2, c) + alphaX * getPixel(x2, y2, c);
                float interpolatedValue = (1 - alphaY) * value1 + alphaY * value2;
                new_img[(y * dst_width + x) * 3 + c] = static_cast<unsigned char>(std::clamp(interpolatedValue, 0.0f, 255.0f));
            }
        }
    }

  delete[] src;
  return new_img;
}


uint uint_min(uint x, int y){
  uint _y = (uint) y;
  if (x>_y)
    return _y;
  return x;
}

double lerp_bli(double c1, double c2, double v1, double v2, double x)
{
  if( (v1==v2) ) return c1;
  double inc = ((c2-c1)/(v2 - v1)) * (x - v1);
  double val = c1 + inc;
  return val;
};

unsigned char *bilinear_resize(unsigned char *src, int height, int width, int dst_height, int dst_width)
{
    unsigned char *new_img = new unsigned char[dst_height*dst_width*3];
    std::memset(new_img, 0, dst_height * dst_width * 3 * sizeof(unsigned char));

    // x and y ratios
    double rx = (double)width / (double)dst_width;
    double ry = (double)height / (double)dst_height;


    
    // loop through destination image
    for(int y=0; y<dst_height; ++y)
    {
        for(int x=0; x<dst_width; ++x)
        {
            //double sx = x * rx;
            //double sy = y * ry;
            double sx = (width>dst_width) ? (x + 0.5) * rx - 0.5 : x * rx;
            double sy = (height>dst_height) ? (y + 0.5) * ry - 0.5 : y * ry;
            
            
            uint xl = std::floor(sx);
            uint yt = std::floor(sy);
            uint xr = (width>dst_width) ? xl+1 : xl;
            uint yb = (height>dst_height) ? yt+1 : yt;
            //uint xr = uint_min(xl + 1, width - 1);
            //uint yb = uint_min(yt + 1, height - 1);


            if (height<dst_height)
              std::cout << yt << ", " << yb << ", " << y << ", " << sy << ". " << height << ", " << dst_height << "\n";

            
            for (uint d = 0; d < 3; ++d)
            {
                unsigned char tl    = src[(xl*width+yt)*3+d];//GetData(xl, yt, d);
                unsigned char tr    = src[(xr*width+yt)*3+d];//GetData(xr, yt, d);
                unsigned char bl    = src[(xl*width+yb)*3+d];//GetData(xl, yb, d);
                unsigned char br    = src[(xr*width+yb)*3+d];//GetData(xr, yb, d);
                double t    = lerp_bli(tl, tr, xl, xr, sx);
                double b    = lerp_bli(bl, br, xl, xr, sx);
                double m    = lerp_bli(t, b, yt, yb, sy);
                unsigned char val   = std::floor(m + 0.5);
                
                new_img[(x * dst_width + y) * 3 + d] = val;
            }
        }
    }

  delete[] src;
  return new_img;
}





