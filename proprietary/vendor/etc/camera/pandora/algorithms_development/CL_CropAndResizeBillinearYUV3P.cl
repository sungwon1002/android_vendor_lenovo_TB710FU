unsigned char BLENDER(int a, int b, int f)
{
  return (unsigned char)((int)(a) + (((int)((f) >> 9) * ((int)(b) - (int)(a)) + 0x40) >> 7));
}

int FixedDiv1_C(int num, int div) {
  return (int)((((long)(num) << 16) - 0x00010001) / (div - 1));
}

int cropInterpolationPlane(
    __global unsigned char *src_ptr,
    __global unsigned char *dst_ptr,
    int k,
    int j,
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight,
    int cropX,
    int cropY,
    int cropW,
    int cropH)
{
    if (2*j >= dstWidth || k >=dstHeight) {
        return 0;
    }

    int dx = FixedDiv1_C(cropW, dstWidth);
    int dy = FixedDiv1_C(cropH, dstHeight);
    int yf = ((cropY << 8) + ((k*dy) >> 8)) & 255;
    int y1_fraction = yf;
    int y0_fraction = 256 - y1_fraction;

    int xi = cropX + ((2*j*dx) >> 16);
    int yi = cropY + ((k*dy) >> 16);
    int a = src_ptr[yi * srcWidth + xi];
    int b = src_ptr[yi * srcWidth + xi + 1];
    long newCropX = ((long)cropX) << 16;
    long newXi = newCropX + (2*j*dx);
    unsigned char curLinex = BLENDER(a, b, newXi & 0xffff);

    xi = cropX + (((2*j+1)*dx) >> 16);
    a = src_ptr[yi * srcWidth + xi];
    b = src_ptr[yi * srcWidth + xi + 1];
    newXi = newCropX + ((2*j+1)*dx);
    unsigned char curLinex1 = BLENDER(a, b, newXi & 0xffff);

    xi = cropX + ((2*j*dx) >> 16);
    yi = cropY + ((k*dy) >> 16) + 1;
    a = src_ptr[yi * srcWidth + xi];
    b = src_ptr[yi * srcWidth + xi + 1];
    newXi = newCropX + (2*j*dx);
    unsigned char nextLinex = BLENDER(a, b, newXi & 0xffff);

    xi = cropX + (((2*j+1)*dx) >> 16);
    yi = cropY + ((k*dy) >> 16) +  1;
    a = src_ptr[yi * srcWidth + xi];
    b = src_ptr[yi * srcWidth + xi + 1];
    newXi = newCropX + ((2*j+1)*dx);
    unsigned char nextLinex1 = BLENDER(a, b, newXi & 0xffff);

    dst_ptr[k*dstWidth+2*j] =
        (curLinex * y0_fraction + nextLinex * y1_fraction + 128) >> 8;
    dst_ptr[k*dstWidth+2*j+1] =
        (curLinex1 * y0_fraction + nextLinex1 * y1_fraction + 128) >> 8;

    return 0;
}

__kernel void CropAndResizeBillinearYUV3P(
    __global unsigned char *srcPtr,
    __global unsigned char *dstPtr,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    int cropX, int cropY,
    int cropW, int cropH)
{
    int j = get_global_id(0);
    int i = get_global_id(1);

    __global unsigned char *yIn = srcPtr;
    __global unsigned char *uIn = srcPtr + srcWidth * srcHeight;
    __global unsigned char *vIn = srcPtr + srcWidth * srcHeight * 5 / 4;

    __global unsigned char *yOut = dstPtr;
    __global unsigned char *uOut = dstPtr + dstWidth * dstHeight;
    __global unsigned char *vOut = dstPtr + dstWidth * dstHeight * 5 / 4;

    cropInterpolationPlane(yIn, yOut, i, j, srcWidth,
        srcHeight, dstWidth, dstHeight,
        cropX, cropY, cropW, cropH);

    cropInterpolationPlane(uIn, uOut, i, j,
        srcWidth/2, srcHeight/2, dstWidth/2, dstHeight/2,
        cropX/2, cropY/2, cropW/2, cropH/2);

    cropInterpolationPlane(vIn, vOut, i, j,
        srcWidth/2, srcHeight/2, dstWidth/2, dstHeight/2,
        cropX/2, cropY/2, cropW/2, cropH/2);
}

