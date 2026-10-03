// CL Kernel Source Code for DeNoiseframeCopy Algorithm
// Version 1.0.0

enum CL_DATA_TYPE_R { BI_BYTE_R, BI_INT_R, BI_FLOAT_R };

__kernel void DeNoiseframeCopy(
    __global const float *input,
    __global float *output,
    int in_id,
    int out_id)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int datalength = width * height;
    __global const float *fptrin = input;
    __global float *fptrout = output;
    *(fptrout + x + y * width + out_id * datalength) = *(fptrin + x + y *width + in_id * datalength);
}
