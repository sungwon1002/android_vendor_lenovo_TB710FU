// CL Kernel Source Code for DefoggingTransmissionLimit Algorithm
// Version 1.0.0

__kernel void DefoggingTransmissionLimit(
    __global const float *image_resize,
    __global float *image_te,
    __global float *output, float A0,
    float A1,
    float A2,
    float K)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    int datalength = width * height;
    float R = image_resize[x + y * width];
    float G = image_resize[x + y * width + datalength];
    float B = image_resize[x + y * width + 2 * datalength];
    float te = image_te[x + y * width];
    float c0 = K / (fabs(R - A0) + 0.0001f);
    float c1 = K / (fabs(G - A1) + 0.0001f);
    float c2 = K / (fabs(B - A2) + 0.0001f);
    float k_t = fmax(fmin(fmin(c0, c1), c2), 1.0f);
    float tl = te * k_t;
    output[x + y * width] = fmin(tl, 1.0f);
}