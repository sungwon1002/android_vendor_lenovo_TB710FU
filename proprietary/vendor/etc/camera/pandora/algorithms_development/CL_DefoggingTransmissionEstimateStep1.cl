// CL Kernel Source Code for DefoggingTransmissionEstimateStep1 Algorithm
// Version 1.0.0

__kernel void DefoggingTransmissionEstimateStep1(
    __global const float *image,
    __global float *output,
    float A0,
    float A1,
    float A2)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    int datalength = width * height;
    output[x + y * width] = image[x + y * width] / A0;
    output[x + y * width + datalength] = image[x + y * width + datalength] / A1;
    output[x + y * width + 2 * datalength] = image[x + y * width + 2 * datalength] / A2;
}



