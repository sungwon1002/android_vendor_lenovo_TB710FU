// CL Kernel Source Code for DefoggingDivision Algorithm
// Version 1.0.0

__kernel void DefoggingDivision(
    __global const float* input1,
    __global const float* input2,
    __global float* output)
{
    int index = get_global_id(0);
    output[index] = input1[index] / input2[index];
}