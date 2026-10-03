// CL Kernel Source Code for nv12ToRGBRemoveStrideScanline Algorithm
// Version 1.0.0

__kernel void nv12ToRGBRemoveStrideScanline(
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

    int strideJ = stride * j;
    int y = pIn[strideJ + i];

    int vOffset = sxc + j / 2 * stride + i - i % 2;
    int u = pIn[vOffset];
    int v = pIn[vOffset + 1];

    int r = y + 1.4075 * (v - 128);
    int g = y - 0.3455 * (u - 128) - 0.7169 * (v - 128);
    int b = y + 1.779 *  (u - 128);

    if (r > 255) r = 255;
    if (g > 255) g = 255;
    if (b > 255) b = 255;
    if (r < 0) r = 0;
    if (g < 0) g = 0;
    if (b < 0) b = 0;

    int widthJ = width * j;
    pOut[widthJ + i] = r;
    pOut[widthJ + i + wxh] = g;
    pOut[widthJ + i + wxh2] = b;
}

