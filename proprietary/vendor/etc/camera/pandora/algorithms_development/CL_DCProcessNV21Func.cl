// CL Kernel Source Code for DCKernelFunc Algorithm
// Version 1.0.0

__kernel void DCProcessNV21Func(
    __global uchar* pInput,
    __global float* preData,
    __global uchar* pOutput,
    int stride, int scanline)
{
    int i   = get_global_id(0);
    int j   = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int datalength = width * height;
    int imageDataLength = stride * scanline;
    float2 xy = vload2(i * 2 + j * width * 4, preData);
    xy.x = xy.x / 2;
    xy.y = xy.y / 2;

    xy.x = min(xy.x,width - 2.0f);
    xy.y = min(xy.y,height - 2.0f);

    int x1 = trunc(xy.x);
    int y1 = trunc(xy.y);
    float kx = xy.x - x1;
    float ky = xy.y - y1;
    float4 uv1 = convert_float4(vload4(0, pInput + (x1 + y1 * stride / 2) * 2 + 1 + imageDataLength))
        * (float4)(1 - kx, 1 - kx, kx, kx);
    float4 uv2 = convert_float4(vload4(0, pInput + (x1 + (y1 + 1) * stride / 2) * 2 + 1 + imageDataLength))
        * (float4)(1 - kx, 1 - kx, kx, kx);
    float4 uv = (1 - ky) * uv1 + ky * uv2;
    pOutput[(i + j * stride / 2) * 2 + 1 + imageDataLength] = trunc(min(max(uv.s0 + uv.s2, 0.0f), 255.0f));
    pOutput[(i + j * stride / 2) * 2 + imageDataLength] = trunc(min(max(uv.s1 + uv.s3, 0.0f), 255.0f));
}
