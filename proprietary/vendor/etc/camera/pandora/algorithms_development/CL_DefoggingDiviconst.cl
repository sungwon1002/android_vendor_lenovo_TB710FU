// CL Kernel Source Code for DefoggingDiviconst Algorithm
// Version 1.0.0

__kernel void DefoggingDiviconst(
    __global float* input,
    float data)
{
   int index = get_global_id(0);
   input[index] = input[index] / data;
}