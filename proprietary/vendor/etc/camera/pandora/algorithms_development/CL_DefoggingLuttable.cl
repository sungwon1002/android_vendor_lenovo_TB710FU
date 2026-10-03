// CL Kernel Source Code for DefoggingLuttable Algorithm
// Version 1.0.0

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

__kernel void DefoggingLuttable(
    __global const float *input,
    __global float *output,
    __global float *table)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    float Y = input[x + y * width];
    int index = (int)Y;
    index = MAX(0, index);
    index = MIN(255, index);
    output[x + y * width] = Y * table[index];
}