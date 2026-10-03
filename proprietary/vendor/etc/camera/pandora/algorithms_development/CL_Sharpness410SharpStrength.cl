// CL Kernel Source Code for Sharpness410SharpStrength Algorithm
// Version 1.0.0

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

__kernel void Sharpness410SharpStrength(
    __global uchar *Y,
    __global float *sharp_img,
    __global float *strength,
    __global float *luma,
    __global float *scope,
    int stride,
    int scanline)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int width  = get_global_size(0);
    const int height = get_global_size(1);
    const int index  = y * width + x;
    float YData = (float)Y[y * stride + x];
    const float sharp_Y  = sharp_img[index];
    float strength_sharp = 0;
    float final_weight   = 0.0f;

    if (YData < luma[0]) {
        strength_sharp = sharp_Y * (strength[0]);
    } else if (YData >= luma[0] && YData < luma[1]) {
        strength_sharp = sharp_Y * (strength[0] + scope[0] * (YData - luma[0]));
    } else if (YData >= luma[1] && YData < luma[2]) {
        strength_sharp = sharp_Y * (strength[1] + scope[1] * (YData - luma[1]));
    } else if (YData >= luma[2]) {
        strength_sharp = sharp_Y * (strength[2] + scope[2] * (YData - luma[2]));
    }
    YData = (YData + strength_sharp);

    Y[y * stride + x] = (uchar)MAX(MIN(YData, 255), 0);
}