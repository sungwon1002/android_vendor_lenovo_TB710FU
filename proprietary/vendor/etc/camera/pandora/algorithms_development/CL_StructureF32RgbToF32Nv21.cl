// CL Kernel Source Code for StructureF32RgbToF32Nv21 Algorithm
// Version 1.0.0

__kernel void StructureF32RgbToF32Nv21(
    __global float *pIn,
    __global float *pOut,
    int stride,
    int scanline)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int frameSize = stride * scanline;
    int frameNoStrideScanlineSize = width * height;
    int planeoffset = width * j + i;
    int nvIndex = j / 2 * stride + i - i % 2;

    float r = pIn[planeoffset];
    float g = pIn[frameNoStrideScanlineSize + planeoffset];
    float b = pIn[frameNoStrideScanlineSize * 2 + planeoffset];

    float y = 0.229f * r + 0.587f * g + 0.114f * b;
    float u = -0.1684f * r - 0.3316f * g + 0.5f * b + 128;
    float v = 0.5f * r - 0.4187f * g - 0.0813f * b + 128;

    if (y > 255) y = 255;
    if (u > 255) u = 255;
    if (v > 255) v = 255;
    if (y < 0) y = 0;
    if (u < 0) u = 0;
    if (v < 0) v = 0;

    pOut[stride * j + i] = y;
    pOut[frameSize + nvIndex + 1] = u;
    pOut[frameSize + nvIndex] = v;
}
