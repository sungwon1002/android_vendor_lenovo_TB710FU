// CL Kernel Source Code for Apply1dLut Algorithm
// Version 1.0.0

__kernel void Apply1dLutUChar(
    __global uchar* input_image,
    __global uchar* output_image,
    __global int* lut,
    float k)
{
    const int gid_x = get_global_id(0);
    const int gid_y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int datalength = width * height;
    const int R = input_image[gid_x + gid_y * width];
    const int G = input_image[gid_x + gid_y * width + datalength];
    const int B = input_image[gid_x + gid_y * width + 2 * datalength];

    output_image[gid_x + gid_y * width] = trunc(min(max(R * (1 - k) + lut[R] * k, 0.0f), 255.0f));
    output_image[gid_x + gid_y * width + datalength] =
        trunc(min(max(G * (1 - k) + lut[G + 256] * k, 0.0f), 255.0f));
    output_image[gid_x + gid_y * width + 2 * datalength] =
        trunc(min(max(B * (1 - k) + lut[B + 512] * k, 0.0f), 255.0f));

}
