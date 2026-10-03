// CL Kernel Source Code for RemoveNoiseYuv2Rgb Algorithm
// Version 1.0.0

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

enum CL_DATA_TYPE_R { BI_BYTE_R, BI_INT_R, BI_FLOAT_R };

__kernel void RemoveNoiseYuv2Rgb(
    __global const float *image,
    __global float *rgb)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int datalength = width * height;
    float Y = image[x + y * width];
    float U = image[x + y * width + datalength];
    float V = image[x + y * width + 2 * datalength];
    float r = Y + 1.4075f * (V - 128);
    float g = Y - 0.3455f * (U - 128) - 0.7169f * (V - 128);
    float b = Y + 1.7790f * (U - 128);
    rgb[x + y * width] = MIN(max(0.0f, r), 255.0f);
    rgb[x + y * width + datalength] = MIN(max(0.0f, g), 255.0f);
    rgb[x + y * width + 2 * datalength] = MIN(max(0.0f, b), 255.0f);
};
