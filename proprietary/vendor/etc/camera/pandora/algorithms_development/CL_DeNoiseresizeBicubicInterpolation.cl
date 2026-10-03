// CL Kernel Source Code for DeNoiseresizeBicubicInterpolation Algorithm
// Version 1.0.0

float cubicWeight(float x)
{
    if (x < 0.0f) {
        x = -x;
    }
    float absX = x;
    float absX2 = absX * absX;
    float absX3 = absX2 * absX;
    float a = -0.5f;

    if (absX <= 1) {
        return (a + 2) * absX3 - (a + 3) * absX2 + 1;
    } else if (absX < 2) {
        return a * absX3 - 5 * a * absX2 + 8 * a * absX - 4 * a;
    }

    return 0;
}

__kernel void DeNoiseresizeBicubicInterpolation(
    __global float *srcPtr,
    __global float *dstPtr,
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight,
    int id)
{
    int j = get_global_id(0);
    int i = get_global_id(1);
    int base1 = srcWidth * srcHeight * id;
    int base2 = dstWidth * dstHeight * id;

    float xRatio = (float)srcWidth / dstWidth;
    float yRatio = (float)srcHeight / dstHeight;
    float weight = 0.0f;
    float sum    = 0.0f;
    int a1, a2, a3, a4;
    int b1, b2, b3, b4;
    int pixel = i * dstWidth + j + base2;
    float w1[4], w2[4];
    float srcy = i * yRatio;
    float srcx = j * xRatio;

    if (srcx - 1 < 0 ) {
        a1 = (int)(srcx + 1);
    } else {
        a1 = (int)(srcx - 1);
    }
    if (srcx < 0) {
        a2 =  0;
    } else {
        a2 = (int)(srcx);
    }
    if (srcx + 1 > srcWidth - 1) {
        a3 = (int)(srcx - 1);
    } else {
        a3 = (int)(srcx + 1);
    }
    if (srcx + 2 > srcWidth - 1) {
        a4 = (int)(srcx - 2);
    } else {
        a4 = (int)(srcx + 2);
    }
    if (srcy - 1 < 0) {
        b1 = (int)(srcy + 1);
    } else {
        b1 = (int)(srcy - 1);
    }
    if (srcy < 0) {
        b2 =  0;
    } else {
        b2 = (int)(srcy);
    }
    if (srcy + 1 > srcHeight - 1) {
        b3 = (int)(srcy - 1);
    } else {
        b3 = (int)(srcy + 1);
    }
    if (srcy + 2 > srcHeight - 1) {
        b4 = (int)(srcy - 2);
    } else {
        b4 = (int)(srcy + 2);
    }
    w1[0] = cubicWeight(srcx - a1);
    w1[1] = cubicWeight(srcx - a2);
    w1[2] = cubicWeight(srcx - a3);
    w1[3] = cubicWeight(srcx - a4);

    w2[0] = cubicWeight(srcy - b1);
    w2[1] = cubicWeight(srcy - b2);
    w2[2] = cubicWeight(srcy - b3);
    w2[3] = cubicWeight(srcy - b4);

    weight = w1[0] * w2[0] + w1[0] * w2[1] + w1[0] * w2[2] + w1[0] * w2[3]
        + w1[1] * w2[0] + w1[1] * w2[1] + w1[1] * w2[2] + w1[1] * w2[3]
        + w1[2] * w2[0] + w1[2] * w2[1] + w1[2] * w2[2] + w1[2] * w2[3]
        + w1[3] * w2[0] + w1[3] * w2[1] + w1[3] * w2[2] + w1[3] * w2[3];
    sum = srcPtr[b1 * srcWidth + a1 + base1] * w1[0] * w2[0]
        + srcPtr[b2 * srcWidth + a1 + base1] * w1[0] * w2[1]
        + srcPtr[b3 * srcWidth + a1 + base1] * w1[0] * w2[2]
        + srcPtr[b4 * srcWidth + a1 + base1] * w1[0] * w2[3]
        + srcPtr[b1 * srcWidth + a2 + base1] * w1[1] * w2[0]
        + srcPtr[b2 * srcWidth + a2 + base1] * w1[1] * w2[1]
        + srcPtr[b3 * srcWidth + a2 + base1] * w1[1] * w2[2]
        + srcPtr[b4 * srcWidth + a2 + base1] * w1[1] * w2[3]
        + srcPtr[b1 * srcWidth + a3 + base1] * w1[2] * w2[0]
        + srcPtr[b2 * srcWidth + a3 + base1] * w1[2] * w2[1]
        + srcPtr[b3 * srcWidth + a3 + base1] * w1[2] * w2[2]
        + srcPtr[b4 * srcWidth + a3 + base1] * w1[2] * w2[3]
        + srcPtr[b1 * srcWidth + a4 + base1] * w1[3] * w2[0]
        + srcPtr[b2 * srcWidth + a4 + base1] * w1[3] * w2[1]
        + srcPtr[b3 * srcWidth + a4 + base1] * w1[3] * w2[2]
        + srcPtr[b3 * srcWidth + a4 + base1] * w1[3] * w2[3];

    dstPtr[pixel] = sum / weight;
}
