// CL Kernel Source Code for DCKernelFunc Algorithm
// Version 1.0.0

__kernel void DCPreProcessFunc(
    __global int *loc,
    __global float *source,
    int x_inter,
    int y_inter,
    __global float *output)
{
    int i   = get_global_id(0);
    int j   = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int n = 0;
    int m = 0;
    float kx1 = 0.0f, ky1 = 0.0f;
    float temp1[2] = { 0.0f, 0.0f };
    float temp2[2] = { 0.0f, 0.0f };
    float x,  y;
    float kx, ky;
    int   x1, y1;
    float R1, R2;
    int datalength = width*height;
    int k = 41;

    n = i / x_inter;
    m = j / y_inter;
    kx1 = (float)(i - loc[(n + m * k) * 2 + 1]) /
        (loc[(n + 1 + m * k) * 2 + 1] - loc[(n + m * k) * 2 + 1]);
    ky1 = (float)(j - loc[(n + m * k) * 2]) /
        (loc[(n + (m + 1) * k) * 2] - loc[(n + m * k) * 2]);
    temp1[0] = (1 - kx1) * source[(n + m * k) * 2+1] +
        kx1 * source[(n + 1 + m * k) * 2 + 1];
    temp1[1] = (1 - kx1) * source[(n + m * k) * 2] +
        kx1 * source[(n + 1 + m * k) * 2];
    temp2[0] = (1 - kx1) * source[(n + (m + 1) * k) * 2 + 1] +
        kx1 * source[((n + 1) + (m + 1) * k) * 2 + 1];
    temp2[1] = (1 - kx1) * source[(n + (m + 1) * k) * 2] +
        kx1 * source[((n + 1) + (m + 1) * k) * 2];
    x = (1 - ky1) * temp1[0] + ky1 * temp2[0];
    y = (1 - ky1) * temp1[1] + ky1 * temp2[1];
    x = fmin(fmax(0.0f, x), width - 2.0f);
    y = fmin(fmax(0.0f, y), height - 2.0f);
    output[(i + j * width) * 2] = x;
    output[(i + j * width) * 2 + 1] = y;
}
