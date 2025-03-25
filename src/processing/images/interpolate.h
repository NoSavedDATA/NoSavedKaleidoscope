#pragma once




unsigned char *interpolate_img(unsigned char *src, int height, int width, int dst_height, int dst_width);

uint uint_min(uint x, int y);

double lerp_bli(double c1, double c2, double v1, double v2, double x);

unsigned char *bilinear_resize(unsigned char *src, int height, int width, int dst_height, int dst_width);



