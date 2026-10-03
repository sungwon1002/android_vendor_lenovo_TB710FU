// CL Kernel Source Code for DefloggingRgb2Yuv Algorithm
// Version 1.0.0

enum CL_DATA_TYPE { BI_BYTE, BI_INT, BI_FLOAT };

__kernel void DefoggingRgb2Yuv(
    __global const float *image,
    __global float *yuv,
    int type)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int datalength = width * height;
    __global const float *fptrin = image;
    __global float *fptrout = yuv;
    float Rf = fptrin[x + y * width];
    float Gf = fptrin[x + y * width + datalength];
    float Bf = fptrin[x + y * width + 2 * datalength];
    fptrout[x + y * width] = (0.299f * Rf + 0.587f * Gf + 0.114f * Bf);
    fptrout[x + y * width + datalength] = (-0.1684f * Rf - 0.3316f * Gf + 0.5f * Bf + 128);
    fptrout[x + y * width + 2 * datalength] = (0.5f * Rf - 0.4187f * Gf - 0.0813f * Bf + 128);
}
