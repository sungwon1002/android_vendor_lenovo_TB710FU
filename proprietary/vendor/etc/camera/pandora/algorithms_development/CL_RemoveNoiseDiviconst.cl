// CL Kernel Source Code for RemoveNoiseDiviconst Algorithm
// Version 1.0.0

__kernel void RemoveNoiseDiviconst(
    __global float* input,
    float data)
{
   int index = get_global_id(0);
   input[index] = input[index] / data;
}
