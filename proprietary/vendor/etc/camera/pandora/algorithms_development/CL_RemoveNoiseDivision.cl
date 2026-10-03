// CL Kernel Source Code for RemoveNoiseDivision Algorithm
// Version 1.0.0

__kernel void RemoveNoiseDivision(
    __global const float* input1,
    __global const float* input2,
    __global float* output,
    int id)
{
    int width = get_global_size(0);
    int height = get_global_size(1);
    int x = get_global_id(0);
    int y = get_global_id(1);
    int index = width * height * id + y *width + x;
    output[index] = input1[index] / input2[index];
}
