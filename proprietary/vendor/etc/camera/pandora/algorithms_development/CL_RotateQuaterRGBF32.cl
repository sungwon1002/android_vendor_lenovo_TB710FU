// CL Kernel Source Code for RotateQuater Algorithm
// Version 1.0.0

typedef enum {
    ROTATE_0,
    ROTATE_90,
    ROTATE_180,
    ROTATE_270,
} RotateType;

__kernel void RotateQuaterRGBF32(
    __global float *input,
    int inWidth,
    int inHeight,
    __global float *output,
    int outWidth,
    int outHeight,
    RotateType type)
{
    int i   = get_global_id(0);
    int j   = get_global_id(1);
    int dim = get_global_id(2);
    int dataSize = inWidth * inHeight;

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

    output[y * outWidth + x + dim * dataSize] = input[j * inWidth + i + dim * dataSize];

}
