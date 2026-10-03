// CL Kernel Source Code for RGBToNv21AddStrideScanline Algorithm
// Version 1.0.0

__kernel void RGBToNv21AddStrideScanline(
    __global unsigned char *pIn,
    __global unsigned char *pOut,
    int width,
    int stride,
    int wxh,
    int wxh2,
    int sxc)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int widthJ = width * j;
    int r = pIn[widthJ + i];
    int g = pIn[widthJ + i + wxh];
    int b = pIn[widthJ + i + wxh2];

    int y = 0.229f * r + 0.587f * g + 0.114f * b;
    int u = -0.1684f * r - 0.3316f * g + 0.5f * b + 128;
    int v = 0.5f * r - 0.4187f * g - 0.0813f * b + 128;

    if (y > 255) y = 255;
    if (u > 255) u = 255;
    if (v > 255) v = 255;
    if (y < 0) y = 0;
    if (u < 0) u = 0;
    if (v < 0) v = 0;

    int strideJ = stride * j;
    pOut[strideJ + i] = y;

    int vOffset = sxc + j / 2 * stride + i - i % 2;
    pOut[vOffset] = v;
    pOut[vOffset + 1] = u;
}

