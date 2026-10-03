// CL Kernel Source Code for DefoggingAddEps Algorithm
// Version 1.0.0

__kernel void DefoggingAddEps(
    __global const float* input,
    __global float* output,
    float eps)
{
    int index = get_global_id(0);
    output[index] = input[index] + eps;
}