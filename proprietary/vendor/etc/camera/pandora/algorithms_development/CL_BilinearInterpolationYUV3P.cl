unsigned char BLENDER(int a, int b, int f)
{
  return (unsigned char)((int)(a) + (((int)((f) >> 9) * ((int)(b) - (int)(a)) + 0x40) >> 7));
}

int FixedDiv1C(int num, int div)
{
    return (int)((((unsigned long)(num) << 16) - 0x00010001) / (div - 1));
}

void InterpolationPlane(
    __global unsigned char *srcPtr,
    __global unsigned char *dstPtr,
    int k,
    int j,
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight)
{
    if (2 * j >= dstWidth || k >= dstHeight) {
        return;
    }

    int dx = FixedDiv1C(srcWidth, dstWidth);
    int dy = FixedDiv1C(srcHeight, dstHeight);

    int yf = ((k * dy) >> 8) & 255;
    int y1Fraction = yf;
    int y0Fraction = 256 - y1Fraction;

    int xi = (2 * j * dx) >> 16;
    int yi = (k * dy) >> 16;
    int a = srcPtr[yi * srcWidth + xi];
    int b = srcPtr[yi * srcWidth + xi + 1];
    unsigned char curLinex = BLENDER(a, b, (2 * j * dx) & 0xffff);

    xi = ((2 * j + 1) * dx) >> 16;
    a = srcPtr[yi * srcWidth + xi];
    b = srcPtr[yi * srcWidth + xi + 1];
    unsigned char curLinex1 = BLENDER(a, b, ((2 * j + 1) * dx) & 0xffff);

    xi = (2 * j * dx) >> 16;
    yi = ((k * dy) >> 16) + 1;
    a = srcPtr[yi * srcWidth + xi];
    b = srcPtr[yi * srcWidth + xi + 1];
    unsigned char nextLinex = BLENDER(a, b, (2 * j * dx) & 0xffff);

    xi = ((2 * j + 1) * dx) >> 16;
    yi = ((k * dy) >> 16) +  1;
    a = srcPtr[yi * srcWidth + xi];
    b = srcPtr[yi * srcWidth + xi + 1];
    unsigned char nextLinex1 = BLENDER(a, b, ((2 * j + 1) * dx) & 0xffff);

    dstPtr[k * dstWidth + 2 * j] =
        (curLinex * y0Fraction + nextLinex * y1Fraction + 128) >> 8;
    dstPtr[k * dstWidth + 2 * j + 1] =
        (curLinex1 * y0Fraction + nextLinex1 * y1Fraction + 128) >> 8;
}

__kernel void BilinearInterpolationYUV3P(
    __global unsigned char *srcPtr,
    __global unsigned char *dstPtr,
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight)
{
    int j = get_global_id(0);
    int i = get_global_id(1);

    __global unsigned char *yIn = srcPtr;
    __global unsigned char *uIn = srcPtr + srcWidth * srcHeight;
    __global unsigned char *vIn = srcPtr + srcWidth * srcHeight * 5 / 4;

    __global unsigned char *yOut = dstPtr;
    __global unsigned char *uOut = dstPtr + dstWidth * dstHeight;
    __global unsigned char *vOut = dstPtr + dstWidth * dstHeight * 5 / 4;

    InterpolationPlane(yIn, yOut, i, j, srcWidth, srcHeight, dstWidth, dstHeight);

    InterpolationPlane(uIn, uOut, i, j, srcWidth/2, srcHeight/2, dstWidth/2, dstHeight/2);

    InterpolationPlane(vIn, vOut, i, j, srcWidth/2, srcHeight/2, dstWidth/2, dstHeight/2);
}

