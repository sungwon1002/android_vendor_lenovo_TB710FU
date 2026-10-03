// CL Kernel Source Code for DefoggingTransmissionEstimateStep2 Algorithm
// Version 1.0.0

__kernel void DefoggingTransmissionEstimateStep2(
    __global const float *image,
    __global float *output,
    float omega)
{
       int x = get_global_id(0);
       int y = get_global_id(1);
       const int width = get_global_size(0);
       const int height = get_global_size(1);
       int datalength = width * height;
       output[x + y * width] = (1 - omega * image[x + y * width]);
 }