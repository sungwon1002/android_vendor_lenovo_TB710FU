void InterAreaResizePlane(
    __global unsigned char *src,
    __global unsigned char *dst,
    int y, int x,
    const int inWidth,
    const int inHeight,
    const int outWidth,
    const int outHeight)
{
    const float scale_x = inWidth * 1.0  / outWidth;
    const float scale_y = inHeight * 1.0 / outHeight;

    const float src_x = x * scale_x;
    const float src_y = y * scale_y;
    const int ix0 = src_x;
    const int iy0 = src_y;
    const int ix1 = min(ix0 + 1, inWidth - 1);
    const int iy1 = min(iy0 + 1, inHeight - 1);
    unsigned char p00 = src[iy0 * inWidth + ix0];
    unsigned char p01 = src[iy0 * inWidth + ix1];
    unsigned char p10 = src[iy1 * inWidth + ix0];
    unsigned char p11 = src[iy1 * inWidth + ix1];
    dst[y * outWidth + x] = (p00 + p01 + p10 + p11) / 4;
}

__kernel void InterAreaResizeYUV3P(
    __global unsigned char *srcPtr,
    __global unsigned char *dstPtr,
    const int srcWidth,
    const int srcHeight,
    const int dstWidth,
    const int dstHeight)
{
    int j = get_global_id(0);
    int i = get_global_id(1);

    __global unsigned char *yIn = srcPtr;
    __global unsigned char *uIn = srcPtr + srcWidth * srcHeight;
    __global unsigned char *vIn = srcPtr + srcWidth * srcHeight * 5 / 4;

    __global unsigned char *yOut = dstPtr;
    __global unsigned char *uOut = dstPtr + dstWidth * dstHeight;
    __global unsigned char *vOut = dstPtr + dstWidth * dstHeight * 5 / 4;

    InterAreaResizePlane(yIn, yOut, i, j, srcWidth, srcHeight, dstWidth, dstHeight);

    InterAreaResizePlane(uIn, uOut, i, j, srcWidth/2, srcHeight/2, dstWidth/2, dstHeight/2);

    InterAreaResizePlane(vIn, vOut, i, j, srcWidth/2, srcHeight/2, dstWidth/2, dstHeight/2);
}

