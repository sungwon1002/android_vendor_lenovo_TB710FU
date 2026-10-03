// CL Kernel Source Code for InterAreaResizeF32RGB Algorithm
// Version 1.0.0

__kernel void InterAreaResizeF32RGB(
    __global float *src,
    __global float *dst,
    const int inWidth,
    const int inHeight,
    const int outWidth,
    const int outHeight)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int dim = get_global_id(2);
    const float scale_x = inWidth * 1.0  / outWidth;
    const float scale_y = inHeight * 1.0 / outHeight;
    const int inSize  = inWidth * inHeight;
    const int outSize = outWidth * outHeight;

    const float src_x = x * scale_x;
    const float src_y = y * scale_y;
    const int ix0 = src_x;
    const int iy0 = src_y;
    const int ix1 = min(ix0 + 1, inWidth - 1);
    const int iy1 = min(iy0 + 1, inHeight - 1);
    float p00 = src[iy0 * inWidth + ix0 + dim * inSize];
    float p01 = src[iy0 * inWidth + ix1 + dim * inSize];
    float p10 = src[iy1 * inWidth + ix0 + dim * inSize];
    float p11 = src[iy1 * inWidth + ix1 + dim * inSize];
    dst[y * outWidth + x + dim * outSize] = (p00 + p01 + p10 + p11) / 4;
}
