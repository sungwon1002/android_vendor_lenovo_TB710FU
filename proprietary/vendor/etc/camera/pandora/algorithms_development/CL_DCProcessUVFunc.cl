// CL Kernel Source Code for DCKernelFunc Algorithm
// Version 1.0.0

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

__kernel void DCProcessUVFunc(
    __global uchar* pInput,
    __global float* preData,
    __global uchar* pOutput)
{
    int i   = get_global_id(0);
    int j   = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int datalength = width * height;
    float x = preData[i * 2 + j * 2 * width * 2] / 2;
    float y = preData[i * 2 + j * 2 * width * 2 + datalength * 4] / 2;

    int x1 = (int)x;
    int y1 = (int)y;
    float kx = x - x1;
    float ky = y - y1;
    float U1 = (1 - kx) * pInput[x1 + y1 * width + datalength * 4] +
        kx * pInput[x1 + 1 + y1 * width + datalength * 4];
    float U2 = (1 - kx) * pInput[x1 + (y1 + 1) * width + datalength * 4] +
        kx * pInput[x1 + 1 + (y1 + 1) * width + datalength * 4];
    pOutput[i + j * width + datalength * 4] = (uchar)(MIN(MAX((1 - ky) * U1 + ky * U2, 0), 255));

    float V1 = (1 - kx) * pInput[x1 + y1 * width + datalength * 5] +
        kx * pInput[x1 + 1 + y1 * width + datalength * 5];
    float V2 = (1 - kx) * pInput[x1 + (y1 + 1) * width + datalength * 5] +
        kx * pInput[x1 + 1 + (y1 + 1) * width + datalength * 5];
    pOutput[i + j * width + datalength * 5] = (uchar)(MIN(MAX((1 - ky) * V1 + ky * V2, 0), 255));
}
