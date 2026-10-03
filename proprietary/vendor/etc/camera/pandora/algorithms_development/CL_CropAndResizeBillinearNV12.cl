int cropBilinearInterpolationPlaneY(
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
    int cropH)
{
    float x_ratio = native_divide((float)cropW, (float)outputWidth);
    float y_ratio = native_divide((float)cropH, (float)outputHeight);
    float inputX = x * x_ratio;
    float inputY = y * y_ratio;
    int x1 = cropX + inputX;
    int y1 = cropY + inputY;
    int x2 = min(x1 + 1, inputWidth - 1);
    int y2 = min(y1 + 1, inputHeight - 1);
    float x_diff = (cropX + inputX) - x1;
    float y_diff = (cropY + inputY) - y1;
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

int cropBilinearInterpolationPlaneU(
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
    int cropH)
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
    int x1 = cropUX + 2 * ((int)(inputX)) + 1;
    int y1 = cropUY + inputY;
    int x2 = min(x1 + 2, inUVWidth - 1);
    int y2 = min(y1 + 1, inUHeight - 1);
    float x_diff = (cropUX + (2.0f * inputX) + 1.0f) - x1;
    float y_diff = (cropUY + inputY) - y1;
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

int cropBilinearInterpolationPlaneV(
    __global uchar *srcVUPtr,
    __global uchar *dstVUPtr,
    int x,
    int y,
    int inVWidth,
    int inVHeight,
    int outVWidth,
    int outVHeight,
    int cropX,
    int cropY,
    int cropW,
    int cropH)
{
    if (x >= outVWidth || y >= outVHeight) {
        return 0;
    }

    int inUVWidth = inVWidth * 2;
    int outUVWidth = outVWidth * 2;

    float x_ratio = native_divide((float)cropW, (float)outVWidth);
    float y_ratio = native_divide((float)cropH, (float)outVHeight);
    float inputX = x * x_ratio;
    float inputY = y * y_ratio;
    int cropVX = cropX / 2 * 2;
    int cropVY = cropY / 2 * 2;
    int x1 = cropVX + 2 * ((int)(inputX));
    int y1 = cropVY + (int)inputY;
    int x2 = min(x1 + 2, inUVWidth - 2);
    int y2 = min(y1 + 1, inVHeight - 1);
    float x_diff = (cropVX + 2.0f * inputX) - x1;
    float y_diff = (cropVY + inputY) - y1;
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

__kernel void CropAndResizeBillinearNV12(
    __global unsigned char *srcPtr,
    __global unsigned char *dstPtr,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    int cropX, int cropY,
    int cropW, int cropH)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    __global unsigned char *yIn = srcPtr;
    __global unsigned char *uvIn = srcPtr + srcWidth * srcHeight;

    __global unsigned char *yOut = dstPtr;
    __global unsigned char *uvOut = dstPtr + dstWidth * dstHeight;

    cropBilinearInterpolationPlaneY(yIn, yOut, x, y,
        srcWidth, srcHeight, dstWidth, dstHeight,
        cropX, cropY, cropW, cropH);

    cropBilinearInterpolationPlaneU(uvIn, uvOut, x, y,
        srcWidth / 2, srcHeight / 2, dstWidth / 2, dstHeight / 2,
        cropX, cropY / 2, cropW / 2, cropH / 2);

    cropBilinearInterpolationPlaneV(uvIn, uvOut, x, y,
        srcWidth / 2, srcHeight / 2, dstWidth / 2, dstHeight / 2,
        cropX, cropY / 2, cropW / 2, cropH / 2);
}
