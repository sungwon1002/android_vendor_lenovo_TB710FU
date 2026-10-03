// CL Kernel Source Code for f32rgbToNv21 Algorithm
// Version 1.0.0

__kernel void f32rgbToNv21(
    __global float *pIn,
    __global unsigned char *pOut)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int frameSize = width * height;
    int planeoffset = width * j + i;

    float r = pIn[planeoffset];
    float g = pIn[planeoffset + frameSize];
    float b = pIn[planeoffset + frameSize * 2];

    float y = 0.229f * r + 0.587f * g + 0.114f * b;
    float u = -0.1684f * r - 0.3316f * g + 0.5f * b + 128;
    float v = 0.5f * r - 0.4187f * g - 0.0813f * b + 128;

    if (y > 255) y = 255;
    if (u > 255) u = 255;
    if (v > 255) v = 255;
    if (y < 0) y = 0;
    if (u < 0) u = 0;
    if (v < 0) v = 0;

    int nv_index = j / 2 * width + i - i % 2;
    pOut[planeoffset] = y;
    pOut[frameSize + nv_index + 1] = u;
    pOut[frameSize + nv_index] = v;
}