// CL Kernel Source Code for StructureF32Nv21ToNv21 Algorithm
// Version 1.0.0

__kernel void StructureF32Nv21ToNv21(
    __global float *pIn,
    __global unsigned char *pOut,
    int stride,
    int scanline)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int frameSize = stride * scanline;
    int planeOffset = j * stride + i;
    int nvIndex = j / 2 * stride + i - i % 2;

    float y = pIn[planeOffset];
    float u = pIn[frameSize + nvIndex + 1];
    float v = pIn[frameSize + nvIndex];

    if (y > 255) y = 255;
    if (u > 255) u = 255;
    if (v > 255) v = 255;
    if (y < 0) y = 0;
    if (u < 0) u = 0;
    if (v < 0) v = 0;

    pOut[planeOffset] = y;
    pOut[frameSize + nvIndex + 1] = u;
    pOut[frameSize + nvIndex] = v;
}