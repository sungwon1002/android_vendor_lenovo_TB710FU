// CL Kernel Source Code for Crop Algorithm
// Version 1.0.0

void cropPlain(
    __global unsigned char *in,  int inputWidth,  int inputHeight,
    __global unsigned char *out, int outputWidth, int outputHeight,
    int  cropX,                  int cropY,       int gidX,
    int gidY)
{
    if (gidX < outputWidth && gidY < outputHeight) {
        int inputX = cropX + gidX;
        int inputY = cropY + gidY;
        if (inputX >= 0 && inputX < inputWidth &&
            inputY >= 0 && inputY < inputHeight) {
            int inputIndex = inputY * inputWidth + inputX;
            int outputIndex = gidY * outputWidth + gidX;
            out[outputIndex] = in[inputIndex];
        }
    }
}

__kernel void CropYUV3P(
    __global unsigned char * srcPtr,
    int srcWidth,
    int srcHeight,
    __global unsigned char * dstPtr,
    int dstWidth,
    int dstHeight,
    int crop_x,
    int crop_y)
{
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);
    int dim = get_global_id(2);
    __global unsigned char *yIn = srcPtr;
    __global unsigned char *uIn = srcPtr + srcWidth * srcHeight;
    __global unsigned char *vIn = srcPtr + srcWidth * srcHeight * 5 / 4;

    __global unsigned char *yOut = dstPtr;
    __global unsigned char *uOut = dstPtr + dstWidth * dstHeight;
    __global unsigned char *vOut = dstPtr + dstWidth * dstHeight * 5 / 4;

    cropPlain(yIn, srcWidth, srcHeight, yOut, dstWidth, dstHeight,
        crop_x, crop_y, gid_x, gid_y);

    cropPlain(uIn, srcWidth / 2, srcHeight / 2, uOut, dstWidth / 2, dstHeight /  2,
        crop_x /  2, crop_y /  2, gid_x, gid_y);

    cropPlain(vIn, srcWidth /  2, srcHeight /  2, vOut, dstWidth /  2, dstHeight /  2,
        crop_x /  2, crop_y /  2, gid_x, gid_y);
}
