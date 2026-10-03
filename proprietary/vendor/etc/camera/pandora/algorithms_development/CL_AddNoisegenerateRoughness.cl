// CL Kernel Source Code for AddNoiseSharpStrength Algorithm
// Version 1.0.0

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

__kernel void AddNoisegenerateRoughness(
    __global uchar *yin,
    __global float *fine,
    __global float *blurred,
    __global float *noisy,
    __global float *blur,
    __global uchar *yout,
   float mWeight)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int width  = get_global_size(0);
    const int height = get_global_size(1);
    const int index  = y * width + x;
    noisy[index] = yin[index] + blurred[index] * 2 * (mWeight) + fine[index] * (1 - mWeight);
    noisy[index] = noisy[index]  < 0.0f ? 0.0f : noisy[index];
    yout[index] = noisy[index] > 255.0f ? 255 : (uchar) noisy[index];
}
