// CL Kernel Source Code for DeNoiseAddSubtract Algorithm
// Version 1.0.0

__kernel void DeNoiseAddSubtract(
    __global const float* input1,
    __global const float* input2,
    __global float* output,
    uchar add , int id)
{
    int width = get_global_size(0);
    int height = get_global_size(1);
    int x = get_global_id(0);
    int y = get_global_id(1);
    int index = y * width + x + width * height * id;
    if (add) {
      output[index] = input1[index] + input2[index];
    } else {
      output[index] = input1[index] - input2[index];
    }
}
