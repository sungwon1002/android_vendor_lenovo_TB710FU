// CL Kernel Source Code for F32RGBToRGB Algorithm
// Version 1.0.0

__kernel void F32RGBToRGB(
    __global float *pIn,
    __global unsigned char *pOut)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int frameSize = width * height;
    int planeOffset = j * width + i;

    int r = pIn[planeOffset];
    int g = pIn[frameSize + planeOffset];
    int b = pIn[frameSize * 2 + planeOffset];

    if (r > 255) r = 255;
    if (g > 255) g = 255;
    if (b > 255) b = 255;
    if (r < 0) r = 0;
    if (g < 0) g = 0;
    if (b < 0) b = 0;

    pOut[planeOffset] = r;
    pOut[frameSize + planeOffset] = g;
    pOut[frameSize * 2 + planeOffset] = b;
}
