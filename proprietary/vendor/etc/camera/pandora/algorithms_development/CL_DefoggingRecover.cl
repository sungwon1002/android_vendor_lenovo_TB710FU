// CL Kernel Source Code for DefoggingRecover Algorithm
// Version 1.0.0

__kernel void DefoggingRecover(
    __global const float* input,
    __global float* input_t,
    __global float* output,
    float A0,
    float A1,
    float A2)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int dim = get_global_id(2);
    const int datalength = width * height;
    output[x + y * width] =(input[x + y * width] - A0) / input_t[x + y * width] + A0;
    output[x + y * width + datalength] =(input[x + y * width + datalength] - A1) / input_t[x + y * width] + A1;
    output[x + y * width + 2 * datalength] =(input[x + y * width + 2 * datalength] - A2) / input_t[x + y * width] + A2;
}