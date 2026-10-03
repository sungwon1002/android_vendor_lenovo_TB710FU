// CL Kernel Source Code for BiCubicInterpolationYUV3P Algorithm
// Version 1.0.0

float cubicWeight(float x)
{
    if (x < 0.0f) {
        x = -x;
    }
    float absX = x;
    float absX2 = absX * absX;
    float absX3 = absX2 * absX;
    float a = -0.5;

    if (absX <= 1) {
        return (a + 2) * absX3 - (a + 3) * absX2 + 1;
    } else if (absX < 2) {
        return a * absX3 - 5 * a * absX2 + 8 * a * absX - 4 * a;
    }

    return 0;
}

__kernel void bicubicInterpolationPlane(
    __global unsigned char *input,
    __global unsigned char *output,
    int i,
    int j,
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight)
{
    if (j >= dstWidth || i >= dstHeight) {
        return;
    }

    float xRatio = (float)srcWidth / dstWidth;
    float yRatio = (float)srcHeight / dstHeight;
    float px = 0.0f, py = 0.0f, dx  = 0.0f, dy = 0.0f, weight = 0.0f;
    float sum   = 0.0f;

    int pixel = i * dstWidth + j;
    px = j * xRatio;
    py = i * yRatio;

    for (dx = -1; dx <= 2; dx++) {
        for (dy = -1; dy <= 2; dy++) {
            int x = (int) (px + dx);
            int y = (int) (py + dy);
            if (x >= 0 && x < srcWidth && y >= 0 && y < srcHeight) {
                weight = cubicWeight(dx) * cubicWeight(dy);
                sum += input[y * srcWidth + x] * weight;
            }
        }
    }

    output[pixel] = sum;
}

__kernel void BiCubicInterpolationYUV3P(
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

    bicubicInterpolationPlane(yIn, yOut, i, j, srcWidth, srcHeight, dstWidth, dstHeight);

    bicubicInterpolationPlane(uIn, uOut, i, j, srcWidth/2, srcHeight/2, dstWidth/2, dstHeight/2);

    bicubicInterpolationPlane(vIn, vOut, i, j, srcWidth/2, srcHeight/2, dstWidth/2, dstHeight/2);
}


