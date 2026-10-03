// CL Kernel Source Code for RemoveNoiseWeigtAddImage Algorithm
// Version 1.0.0

__kernel void RemoveNoiseWeigtAddImage(
    __global float *input_image1,
    __global float *input_image2,
    __global float *output_image,
    float m, int id)
{
    int width = get_global_size(0);
    int height = get_global_size(1);
    int x = get_global_id(0);
    int y = get_global_id(1);
    int base = width * height * id;
    output_image[y * width + x + base] = input_image1[y * width + x + base] * m + input_image2[y * width + x + base] * (1 - m);
}
