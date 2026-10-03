int bilinearInterpolationPlaneY(
    __global uchar *srcPtr,
    __global uchar *dstPtr,
    int x,
    int y,
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight)
{
    float x_ratio = native_divide((float)inputWidth, (float)outputWidth);
    float y_ratio = native_divide((float)inputHeight, (float)outputHeight);
    float inputX = x * x_ratio;
    float inputY = y * y_ratio;
    int x1 = trunc(inputX);
    int y1 = trunc(inputY);
    int x2 = min(x1 + 1, inputWidth - 1);
    int y2 = min(y1 + 1, inputHeight - 1);
    float x_diff = inputX - x1;
    float y_diff = inputY - y1;
    uchar pixel1 = srcPtr[y1 * inputWidth + x1];
    uchar pixel2 = srcPtr[y1 * inputWidth + x2];
    uchar pixel3 = srcPtr[y2 * inputWidth + x1];
    uchar pixel4 = srcPtr[y2 * inputWidth + x2];
    float top_pixel = pixel1 * (1 - x_diff) + pixel2 * x_diff;
    float bottom_pixel = pixel3 * (1 - x_diff) + pixel4 * x_diff;
    dstPtr[y * outputWidth + x] =
        top_pixel * (1 - y_diff) + bottom_pixel * y_diff;

    return 0;
}

int bilinearInterpolationPlaneU(
    __global uchar *srcVUPtr,
    __global uchar *dstVUPtr,
    int x,
    int y,
    int inUWidth,
    int inUHeight,
    int outUWidth,
    int outUHeight)
{
    if (x >= outUWidth || y >= outUHeight) {
        return 0;
    }

    int inUVWidth = inUWidth * 2;
    int outUVWidth = outUWidth * 2;

    float x_ratio = native_divide((float)inUWidth, (float)outUWidth);
    float y_ratio = native_divide((float)inUHeight, (float)outUHeight);
    float inputX = x * x_ratio;
    float inputY = y * y_ratio;
    int x1 = 2 * trunc(inputX) + 1;
    int y1 = trunc(inputY);
    int x2 = min(x1 + 2, inUVWidth - 1);
    int y2 = min(y1 + 1, inUHeight - 1);
    float x_diff = ((2.0f * inputX) + 1.0f) - x1;
    float y_diff = inputY - y1;
    uchar pixel1 = srcVUPtr[y1 * inUVWidth + x1];
    uchar pixel2 = srcVUPtr[y1 * inUVWidth + x2];
    uchar pixel3 = srcVUPtr[y2 * inUVWidth + x1];
    uchar pixel4 = srcVUPtr[y2 * inUVWidth + x2];
    float top_pixel = pixel1 * (1 - x_diff) + pixel2 * x_diff;
    float bottom_pixel = pixel3 * (1 - x_diff) + pixel4 * x_diff;
    dstVUPtr[y * outUVWidth + (2 * x + 1)] =
        top_pixel * (1 - y_diff) + bottom_pixel * y_diff;

    return 0;
}

int bilinearInterpolationPlaneV(
    __global uchar *srcVUPtr,
    __global uchar *dstVUPtr,
    int x,
    int y,
    int inVWidth,
    int inVHeight,
    int outVWidth,
    int outVHeight)
{
    if (x >= outVWidth || y >= outVHeight) {
        return 0;
    }

    int inUVWidth = inVWidth * 2;
    int outUVWidth = outVWidth * 2;

    float x_ratio = native_divide((float)inVWidth, (float)outVWidth);
    float y_ratio = native_divide((float)inVHeight, (float)outVHeight);
    float inputX = x * x_ratio;
    float inputY = y * y_ratio;
    int x1 = 2 * trunc(inputX);
    int y1 = trunc(inputY);
    int x2 = min(x1 + 2, inUVWidth - 2);
    int y2 = min(y1 + 1, inVHeight - 1);
    float x_diff = (2.0f * inputX) - x1;
    float y_diff = inputY - y1;
    uchar pixel1 = srcVUPtr[y1 * inUVWidth + x1];
    uchar pixel2 = srcVUPtr[y1 * inUVWidth + x2];
    uchar pixel3 = srcVUPtr[y2 * inUVWidth + x1];
    uchar pixel4 = srcVUPtr[y2 * inUVWidth + x2];
    float top_pixel = pixel1 * (1 - x_diff) + pixel2 * x_diff;
    float bottom_pixel = pixel3 * (1 - x_diff) + pixel4 * x_diff;
    dstVUPtr[y * outUVWidth + 2 * x] =
        top_pixel * (1 - y_diff) + bottom_pixel * y_diff;

    return 0;
}

__kernel void BilinearInterpolationNV12(
    __global uchar *srcPtr,
    __global uchar *dstPtr,
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    __global uchar *yIn = srcPtr;
    __global uchar *uvIn = srcPtr + srcWidth * srcHeight;

    __global uchar *yOut = dstPtr;
    __global uchar *uvOut = dstPtr + dstWidth * dstHeight;

    bilinearInterpolationPlaneY(yIn, yOut, x, y, srcWidth, srcHeight, dstWidth, dstHeight);
    bilinearInterpolationPlaneU(uvIn, uvOut, x, y,
        srcWidth / 2, srcHeight / 2, dstWidth / 2, dstHeight / 2);
    bilinearInterpolationPlaneV(uvIn, uvOut, x, y,
        srcWidth / 2, srcHeight / 2, dstWidth / 2, dstHeight / 2);
}
