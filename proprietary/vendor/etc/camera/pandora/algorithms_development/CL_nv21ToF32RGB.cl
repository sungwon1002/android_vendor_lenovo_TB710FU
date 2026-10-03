// CL Kernel Source Code for nv21ToF32RGB Algorithm
// Version 1.0.0

__kernel void nv21ToF32RGB(
    __global unsigned char *pIn,
    __global float *pOut)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int frameSize = width * height;
    int planeOffset = j * width + i;
    int nvIndex = j / 2 * width + i - i % 2;

    float y = pIn[planeOffset];
    float u = pIn[frameSize + nvIndex + 1];
    float v = pIn[frameSize + nvIndex];

    float r = y + 1.4075f * (v - 128);
    float g = y - 0.3455f * (u - 128) - 0.7169f * (v - 128);
    float b = y + 1.779f *  (u - 128);

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