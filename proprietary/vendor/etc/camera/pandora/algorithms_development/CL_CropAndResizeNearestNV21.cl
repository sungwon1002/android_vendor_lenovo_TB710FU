int cropNearestInterpolationPlaneY(
    __global uchar *srcPtr,
    __global uchar *dstPtr,
    int x,
    int y,
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight,
    int cropX,
    int cropY,
    int cropW,
    int cropH,
    int stride)
{
    float x_ratio = native_divide((float)cropW, (float)outputWidth);
    float y_ratio = native_divide((float)cropH, (float)outputHeight);

    float inputX = x * x_ratio;
    float inputY = y * y_ratio;
    int srcX = (int)(cropX + inputX + 0.5f);
    int srcY = (int)(cropY + inputY + 0.5f);
    dstPtr[y * stride + x] =
        srcPtr[srcY * stride + srcX];

    return 0;
}

int cropNearestInterpolationPlaneUV(
    __global uchar *srcVUPtr,
    __global uchar *dstVUPtr,
    int x,
    int y,
    int inUWidth,
    int inUHeight,
    int outUWidth,
    int outUHeight,
    int cropX,
    int cropY,
    int cropW,
    int cropH,
    int stride)
{
    if (x >= outUWidth || y >= outUHeight) {
        return 0;
    }

    int inUVWidth = inUWidth * 2;
    int outUVWidth = outUWidth * 2;

    float x_ratio = native_divide((float)cropW, (float)outUWidth);
    float y_ratio = native_divide((float)cropH, (float)outUHeight);

    float inputX = x * x_ratio;
    float inputY = y * y_ratio;
    int cropUX = cropX / 2 * 2;
    int cropUY = cropY / 2 * 2;
    //U
    int srcXU = cropUX + (((int)(2 * inputX + 1 + 0.5f)) / 2 * 2) + 1;
    int srcYU = (int)(cropUY + inputY + 0.5f);
    dstVUPtr[y * stride + (2 * x + 1)] =
        srcVUPtr[srcYU * stride + srcXU];
    //V
    int srcXV = cropUX + (((int)(2 * inputX + 0.5f)) / 2 * 2);
    int srcYV = (int)(cropUY + inputY + 0.5f);
    dstVUPtr[y * stride + 2 * x] =
        srcVUPtr[srcYV * stride + srcXV];

    return 0;
}

__kernel void CropAndResizeNearestNV21(
    __global unsigned char *srcPtr,
    __global unsigned char *dstPtr,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    int cropX, int cropY,
    int cropW, int cropH,
    int stride, int scanline)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    __global unsigned char *yIn = srcPtr;
    __global unsigned char *uvIn = srcPtr + stride * scanline;

    __global unsigned char *yOut = dstPtr;
    __global unsigned char *uvOut = dstPtr + stride * scanline;

    cropNearestInterpolationPlaneY(yIn, yOut, x, y,
        srcWidth, srcHeight, dstWidth, dstHeight,
        cropX, cropY, cropW, cropH, stride);

    cropNearestInterpolationPlaneUV(uvIn, uvOut, x, y,
        srcWidth / 2, srcHeight / 2, dstWidth / 2, dstHeight / 2,
        cropX, cropY / 2, cropW / 2, cropH / 2, stride);
}
