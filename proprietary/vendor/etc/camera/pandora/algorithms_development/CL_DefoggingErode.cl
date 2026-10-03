// CL Kernel Source Code for DefoggingErode Algorithm
// Version 1.0.0

enum CL_DATA_TYPE { BI_BYTE, BI_INT, BI_FLOAT };

__kernel void DefoggingErode(
    __global const float* image,
    __global float* output,
    int kernel_size,
    int type)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int datalength = width * height;
    uchar umin = 255;
    float fmin = 255.0f;
    for (int i = -kernel_size / 2; i <= kernel_size / 2; i++) {
      for (int j = -kernel_size / 2; j <= kernel_size / 2; j++) {
        const int px = clamp(x + i, 0, width - 1);
        const int py = clamp(y + j, 0, height - 1);
        float frel = image[py * width + px];
        fmin = (fmin < frel) ? fmin : frel;
      }
    }

    output[x + y * width] = fmin;
}