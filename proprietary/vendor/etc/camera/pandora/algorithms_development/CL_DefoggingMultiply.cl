// CL Kernel Source Code for DefoggingMultiply Algorithm
// Version 1.0.0

__kernel void DefoggingMultiply(
    __global const float* input1,
    __global const float* input2,
    __global float* output)
{
    int index = get_global_id(0);
    output[index] = input1[index] * input2[index];
};