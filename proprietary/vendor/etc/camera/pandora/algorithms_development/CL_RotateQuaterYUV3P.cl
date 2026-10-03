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
    int inWidth,
    int inHeight,
    __global unsigned char *output,
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

    output[y * outWidth + x] = input[j * inWidth + i];
}


__kernel void RotateQuaterYUV3P(
    __global unsigned char *srcPtr,
    int srcWidth,
    int srcHeight,
    __global unsigned char *dstPtr,
    int dstWidth,
    int dstHeight,
    RotateType type)
{
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);

    __global unsigned char *yIn = srcPtr;
    __global unsigned char *uIn = yIn + srcWidth * srcHeight;
    __global unsigned char *vIn = yIn + (srcWidth * srcHeight) * 5 / 4;

    __global unsigned char *yOut = dstPtr;
    __global unsigned char *uOut = yOut + dstWidth * dstHeight;
    __global unsigned char *vOut = yOut + (dstWidth * dstHeight) * 5 / 4;

    RotateQuaterPlain(yIn, srcWidth, srcHeight,
        yOut, dstWidth, dstHeight, gid_x, gid_y, type);

    if (gid_x % 2 == 0 && gid_y % 2 == 0) {
        RotateQuaterPlain(uIn, srcWidth / 2, srcHeight / 2 ,
            uOut, dstWidth / 2 , dstHeight / 2, gid_x / 2, gid_y / 2, type);

        RotateQuaterPlain(vIn, srcWidth / 2, srcHeight / 2,
            vOut, dstWidth / 2 , dstHeight / 2, gid_x / 2, gid_y / 2, type);
    }
}
