// CL Kernel Source Code for RotateQuater Algorithm
// Version 1.0.0

typedef enum {
    ROTATE_0,
    ROTATE_90,
    ROTATE_180,
    ROTATE_270,
} RotateType;

void RotateQuaterPlain(
    __global unsigned char *input,
    int inStride,
    int inScanline,
    int inWidth,
    int inHeight,
    __global unsigned char *output,
    int outStride,
    int outScanline,
    int outWidth,
    int outHeight,
    int i,
    int j,
    RotateType type)
{
    int x;
    int y;

    switch(type){
        case ROTATE_0:
            x = i;
            y = j;
            break;
        case ROTATE_90:
            x = inHeight - j - 1;
            y = i;
            break;
        case ROTATE_180:
            x = inWidth - i - 1;
            y = inHeight - j - 1;
            break;
        case ROTATE_270:
            x = j;
            y = inWidth - i - 1;
            break;
    }

    *(output + y * outStride + x) = *(input + j * inStride + i);
}

void RotateQuaterPlain2Byte(
    __global unsigned short int *input,
    int inStride,
    int inScanline,
    int inWidth,
    int inHeight,
    __global unsigned short int *output,
    int outStride,
    int outScanline,
    int outWidth,
    int outHeight,
    int i,
    int j,
    RotateType type)
{
    int x;
    int y;

    switch(type){
        case ROTATE_0:
            x = i;
            y = j;
            break;
        case ROTATE_90:
            x = inHeight - j - 1;
            y = i;
            break;
        case ROTATE_180:
            x = inWidth - i - 1;
            y = inHeight - j - 1;
            break;
        case ROTATE_270:
            x = j;
            y = inWidth - i - 1;
            break;
    }

    output[y * outStride + x] = input[j * inStride + i];
}

__kernel void RotateQuaterYUV2P(
    __global unsigned char *srcPtr,
    int srcStride,
    int srcScanline,
    int srcWidth,
    int srcHeight,
    __global unsigned char *dstPtr,
    int dstStride,
    int dstScanline,
    int dstWidth,
    int dstHeight,
    RotateType type)
{
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);

    __global unsigned char *yIn = srcPtr;
    __global unsigned short int *uVIn = (__global unsigned short int *)(yIn + srcStride * srcScanline);

    __global unsigned char *yOut = dstPtr;
    __global unsigned short int *uVOut = (__global unsigned short int *)(yOut + dstStride * dstScanline);

    RotateQuaterPlain(yIn,  srcStride, srcScanline, srcWidth, srcHeight,
        yOut, dstStride, dstScanline, dstWidth, dstHeight,
        gid_x, gid_y, type);

    if (gid_y % 2 == 0) {
        RotateQuaterPlain2Byte(uVIn, srcStride / 2, srcScanline / 2 , srcWidth / 2, srcHeight / 2,
            uVOut, dstStride / 2, dstScanline / 2, dstWidth / 2, dstHeight / 2,
            gid_x, gid_y / 2, type);
    }
}
