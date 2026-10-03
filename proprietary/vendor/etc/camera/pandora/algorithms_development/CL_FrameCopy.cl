// CL Kernel Source Code for FrameCopy Algorithm
// Version 1.0.0

__kernel void FrameCopy(
    __global uchar *input,
    __global uchar *output)
{
    int index = get_global_id(0);
    output[index] = input[index];
}
