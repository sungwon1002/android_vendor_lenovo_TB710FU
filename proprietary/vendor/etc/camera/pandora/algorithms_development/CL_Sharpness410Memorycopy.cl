// CL Kernel Source Code for Sharpness410Convolution Algorithm
// Version 1.0.0

__kernel void Sharpness410Memorycopy(
    __global uchar *input,
    __global uchar *output)
{
    int index = get_global_id(0);
    output[index] = input[index];
}
