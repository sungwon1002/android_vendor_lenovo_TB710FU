// CL Kernel Source Code for DeNoiseBilinear Algorithm
// Version 1.0.0

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

__kernel void DeNoiseBilinear(
    __global float* input,
    __global float* output,
    int width,int height,
    float scale_factor,
    int id)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int newStride = get_global_size(0);
    int newScanline = get_global_size(1);
    const int datalength = width * height;
    const int newDataLength = newStride * newScanline;
    __global float* fptrin = NULL;
    __global float* fptrout = NULL;
    float u = (float)i * (float)(1.0f / scale_factor);
    float v = (float)j * (float)(1.0f / scale_factor);
    int x = (int)u;
    int y = (int)v;
    float dx = u - x;
    float dy = v - y;
    int index = j * newStride + i;
    int x_inc = MIN(x + 1, width - 1);
    int y_inc = MIN(y + 1, height - 1);
    if (i < newStride && j < newScanline) {
      fptrin = input;
      fptrout = output;
      fptrout[index + id * newDataLength] =
        (fptrin[y * width + x + id * datalength] * (1.0f - dx)*(1.0f - dy) +
        fptrin[y_inc * width + x + id * datalength] * (1.0f - dx) * dy +
        fptrin[y * width + x_inc + id * datalength] * dx * (1.0f - dy) +
        fptrin[y_inc * width + x_inc + id * datalength] * dx * dy);
    }
}
