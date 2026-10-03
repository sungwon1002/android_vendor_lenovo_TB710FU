// CL Kernel Source Code for yuvToRGB Algorithm
// Version 1.0.0

__kernel void yuvToRGB(
    __global unsigned char *pIn,
    __global unsigned char *pOut)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int planeOffset = j * width + i;
    int frameSize = width * height;
    int uvOffset = (j / 2) * (width / 2) + i / 2;
    int uIndex = frameSize + uvOffset;
    int vIndex = uIndex + frameSize / 4;

    int y = pIn[planeOffset];
    int u = pIn[uIndex];
    int v = pIn[vIndex];

    int r = y + 1.4075f * (v - 128);
    int g = y - 0.3455f * (u - 128) - 0.7169f * (v - 128);
    int b = y + 1.779f *  (u - 128);

    if (r > 255) r = 255;
    if (g > 255) g = 255;
    if (b > 255) b = 255;
    if (r < 0) r = 0;
    if (g < 0) g = 0;
    if (b < 0) b = 0;

    pOut[planeOffset] = r;
    pOut[planeOffset + frameSize] = g;
    pOut[planeOffset + frameSize * 2] = b;
}