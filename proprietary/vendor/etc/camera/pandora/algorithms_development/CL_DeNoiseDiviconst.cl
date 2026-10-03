// CL Kernel Source Code for DeNoiseDiviconst Algorithm
// Version 1.0.0

__kernel void DeNoiseDiviconst(
    __global float* input,
    float data)
{
   int index = get_global_id(0);
   input[index] = input[index] / data;
}
