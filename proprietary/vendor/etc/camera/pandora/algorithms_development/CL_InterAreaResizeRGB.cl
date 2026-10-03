// CL Kernel Source Code for InterAreaResizeRGB Algorithm
// Version 1.0.0

__kernel void InterAreaResizeRGB(
    __global unsigned char *src,
    __global unsigned char *dst,
    int inWidth,
    int inHeight,
    int outWidth,
    int outHeight)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int dim = get_global_id(2);
    float scale_x = inWidth * 1.0  / outWidth;
    float scale_y = inHeight * 1.0 / outHeight;
    int inSize  = inWidth * inHeight;
    int outSize = outWidth * outHeight;

    float src_x = x * scale_x;
    float src_y = y * scale_y;
    int ix0 = src_x;
    int iy0 = src_y;
    int ix1 = min(ix0 + 1, inWidth - 1);
    int iy1 = min(iy0 + 1, inHeight - 1);
    int p00 = src[iy0 * inWidth + ix0 + dim * inSize];
    int p01 = src[iy0 * inWidth + ix1 + dim * inSize];
    int p10 = src[iy1 * inWidth + ix0 + dim * inSize];
    int p11 = src[iy1 * inWidth + ix1 + dim * inSize];
    dst[y * outWidth + x + dim * outSize] = (p00 + p01 + p10 + p11) / 4;
}
