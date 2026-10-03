// CL Kernel Source Code for StructureF32Nv21ToF32Rgb Algorithm
// Version 1.0.0

__kernel void StructureF32Nv21ToF32Rgb(
    __global float *pIn,
    __global float *pOut,
    int stride,
    int scanline)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int planeOffset = j * width + i;
    int frameSize = stride * scanline;
    int frameNoStrideScanlineSize = width * height;
    int nvIndex = j / 2 * stride + i - i % 2;

    float y = pIn[j * stride + i];
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
    pOut[frameNoStrideScanlineSize + planeOffset] = g;
    pOut[frameNoStrideScanlineSize * 2 + planeOffset] = b;
}
