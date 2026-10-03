// CL Kernel Source Code for HDRCheckScaleCvt Algorithm
// Version 1.0.0

__kernel void HDRCheckScaleCvt(
    __global unsigned char *pInputYuv,
    __global int *hist,
    __global unsigned char *pOutputGray,
    __private const int step,
    __private const int fullStride)
{
    int j = get_global_id(0);
    int i = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int index  = i * width + j;

    pOutputGray[index] = pInputYuv[step * (i * fullStride + j)];
    int val = pOutputGray[index];
    atomic_inc(&hist[val]);
}
