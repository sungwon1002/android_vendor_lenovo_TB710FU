// CL Kernel Source Code for DefoggingHistogram Algorithm
// Version 1.0.0

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

__kernel void DefoggingHistogram(
    __global const float *image,
    __global int *hist)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    if (x < width && y < height) {
        int bin = (int)image[y * width + x];
        bin = MAX(min(255, bin), 0);
        atomic_add(&hist[bin], 1);
    }
}