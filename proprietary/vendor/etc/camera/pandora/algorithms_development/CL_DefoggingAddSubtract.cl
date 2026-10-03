// CL Kernel Source Code for DefoggingAddSubtract Algorithm
// Version 1.0.0

__kernel void DefoggingAddSubtract(
    __global const float* input1,
    __global const float* input2,
    __global float* output,
    uchar add)
{
    int index = get_global_id(0);
    if (add) {
      output[index] = input1[index] + input2[index];
    } else {
      output[index] = input1[index] - input2[index];
    }
}