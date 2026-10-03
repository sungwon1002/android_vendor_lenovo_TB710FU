// CL Kernel Source Code for DCKernelFunc Algorithm
// Version 1.0.0

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

__kernel void DCProcessYFunc(
    const __global uchar* pInput,
    __global float* preData,
    __global uchar* pOutput,
    int stride,
    int scanline)
{
    int i   = get_global_id(0);
    int j   = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);

    float2 xy = vload2(i + j * width, preData);

    int x1 = (int)xy.x;
    int y1 = (int)xy.y;
    float kx = xy.x - x1;
    float ky = xy.y - y1;

    float2 R1 = convert_float2(vload2(0, pInput + x1 + y1 * stride )) * (float2)(1 - kx, kx);

    float2 R2 = convert_float2(vload2(0, pInput + x1 + (y1 + 1) * stride)) * (float2)(1 - kx, kx);

    float2 result = (1 - ky) * R1 + ky * R2;

    pOutput[i + j * stride] = trunc(min(max(result.x + result.y, 0.0f), 255.0f));
}
