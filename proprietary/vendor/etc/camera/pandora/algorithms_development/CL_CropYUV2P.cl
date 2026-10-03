// CL Kernel Source Code for Crop Algorithm
// Version 1.0.0

void cropPlain(
    __global unsigned char *in,  int inputWidth,  int inputHeight,
    __global unsigned char *out, int outputWidth, int outputHeight,
    int cropX, int cropY, int x, int y, int inStride, int outStride)
{
    out[y * outStride + x] = in[(cropY + y) * inStride + (cropX + x)];
}

void cropUVPlain(
    __global unsigned char *in,  int inputWidth,  int inputHeight,
    __global unsigned char *out, int outputWidth, int outputHeight,
    int  cropX, int cropY, int x, int y, int inStride, int outStride)
{
    if (y >= outputHeight) {
        return;
    }

    int alignCropX = cropX / 2 * 2;
    out[y * outStride + x] = in[(cropY + y) * inStride + (alignCropX + x)];
}

__kernel void CropYUV2P(
    __global unsigned char * srcPtr,
    int srcWidth,
    int srcHeight,
    __global unsigned char * dstPtr,
    int dstWidth,
    int dstHeight,
    int crop_x,
    int crop_y,
    int inStride,
    int inScanline,
    int outStride,
    int outScanline)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    __global unsigned char *yIn = srcPtr;
    __global unsigned char *uVIn = srcPtr + inStride * inScanline;

    __global unsigned char *yOut = dstPtr;
    __global unsigned char *uVOut = dstPtr + outStride * outScanline;

    cropPlain(yIn, srcWidth, srcHeight, yOut, dstWidth, dstHeight,
        crop_x, crop_y, x, y, inStride, outStride);

    cropUVPlain(uVIn, srcWidth, srcHeight / 2, uVOut, dstWidth, dstHeight / 2,
        crop_x, crop_y / 2, x, y, inStride, outStride);
}

