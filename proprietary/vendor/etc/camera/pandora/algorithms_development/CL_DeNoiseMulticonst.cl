// CL Kernel Source Code for DeNoiseMulticonst Algorithm
// Version 1.0.0

__kernel void DeNoiseMulticonst(
    __global float *input_image,
    float data, int id)
{
    int width = get_global_size(0);
    int height = get_global_size(1);
    int x = get_global_id(0);
    int y = get_global_id(1);
    int index = y * width + x + width * height * id;
    input_image[index] *= data;
}
