// CL Kernel Source Code for Rgb2Hls Algorithm
// Version 1.0.0
#define FORMAT_RGB    9
#define  DATA_H              31
#define  DATA_W              41
#define ARRAY_X(x, a, b, c)  *((x) + ((a) * DATA_W + (b)) * 2 + (c))
#define MAX(a, b)            ((a) > (b) ? (a) : (b))
#define MIN(a, b)            ((a) < (b) ? (a) : (b))
#define CLIP(x,a,b)          (MIN(MAX(x,a),b))

void AtomicAdd(volatile __global float *source, const float operand)
{
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source,
                             prevVal.intVal, newVal.intVal)
                             != prevVal.intVal);
    return;
}

__kernel void DistortionStep1(
    __global void *pIn,
    __global float *mSource,
    __global int *mGridX,
    __global int *mGridY,
    __global float *mKX,
    __global float *mKY,
             float k,
    __global float *n_val,
    __global float *pOut,
             int format)
{
    int j = get_global_id(0);
    int i = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int frameSize = width * height;
    float r = 0;
    float g = 0;
    float b = 0;
    if(format == FORMAT_RGB){
        __global uchar * pTemp = (__global uchar *)pIn;
        r = (float)pTemp[width * i + j];
        g = (float)pTemp[width * i + j + frameSize];
        b = (float)pTemp[width * i + j + frameSize * 2];
    } else {
        __global float * pTemp = (__global float *)pIn;
        r = pTemp[width * i + j];
        g = pTemp[width * i + j + frameSize];
        b = pTemp[width * i + j + frameSize * 2];
    }
    float temp1[2] = {0, 0};
    float temp2[2] = {0, 0};
    float temp[2]  = {0, 0};
    float x  = 0, y  = 0;
    int   x1 = 0, y1 = 0;
    float w1 = 0, w2 = 0, w3 = 0, w4 = 0;
    int   temp_h = height + 3;
    int   temp_w = width + 3;
    __global float *img1  = pOut;
    __global float *img1R = img1;
    __global float *img1G = (img1 + temp_w * temp_h);
    __global float *img1B = (img1 + temp_w * temp_h * 2);
    __global float *n = n_val;

    temp1[0] = (1 - mKX[i]) * ARRAY_X(mSource, mGridX[i], mGridY[j], 0)
        + mKX[i] * ARRAY_X(mSource, mGridX[i] + 1, mGridY[j], 0);
    temp1[1] = (1 - mKX[i]) * ARRAY_X(mSource, mGridX[i], mGridY[j], 1)
        + mKX[i] * ARRAY_X(mSource, mGridX[i] + 1, mGridY[j], 1);
    temp2[0] = (1 - mKX[i]) * ARRAY_X(mSource, mGridX[i], mGridY[j] + 1, 0)
        + mKX[i] * ARRAY_X(mSource, mGridX[i] + 1, mGridY[j] + 1, 0);
    temp2[1] = (1 - mKX[i]) * ARRAY_X(mSource, mGridX[i], mGridY[j] + 1, 1)
        + mKX[i] * ARRAY_X(mSource, mGridX[i] + 1, mGridY[j] + 1, 1);
    temp[0] =(1 - mKY[j]) * temp1[0] + mKY[j] * temp2[0];
    temp[1] =(1 - mKY[j]) * temp1[1] + mKY[j] * temp2[1];
    x = k * temp[0] + (1 - k) * i + 1;
    y = k * temp[1] + (1 - k) * j + 1;

    if (x < 0 || y < 0 || x > height + 1 || y > width + 1) {
        return;
    }

    x = CLIP(x, 0, height + 1);
    y = CLIP(y, 0, width + 1);

    x1 = (int) x;
    y1 = (int) y;
    w1 = (1 - (x - x1)) * (1-(y-y1));
    w2 = (1 - (x - x1)) * (y-y1);
    w3 = (x - x1) * (1 - (y - y1));
    w4 = (x - x1) * (y - y1);

    AtomicAdd((volatile __global float *)(img1R + x1 * temp_w + y1), r * w1);
    AtomicAdd((volatile __global float *)(img1G + x1 * temp_w + y1), g * w1);
    AtomicAdd((volatile __global float *)(img1B + x1 * temp_w + y1), b * w1);
    AtomicAdd((volatile __global float *)(img1R + x1 * temp_w + (y1 + 1)), r * w2);
    AtomicAdd((volatile __global float *)(img1G + x1 * temp_w + (y1 + 1)), g * w2);
    AtomicAdd((volatile __global float *)(img1B + x1 * temp_w + (y1 + 1)), b * w2);

    AtomicAdd((volatile __global float *)(img1R + (x1 + 1) * temp_w + y1), r * w3);
    AtomicAdd((volatile __global float *)(img1G + (x1 + 1) * temp_w + y1), g * w3);
    AtomicAdd((volatile __global float *)(img1B + (x1 + 1) * temp_w + y1), b * w3);
    AtomicAdd((volatile __global float *)(img1R + (x1 + 1) * temp_w + (y1 + 1)), r * w4);
    AtomicAdd((volatile __global float *)(img1G + (x1 + 1) * temp_w + (y1 + 1)), g * w4);
    AtomicAdd((volatile __global float *)(img1B + (x1 + 1) * temp_w + (y1 + 1)), b * w4);

    AtomicAdd((volatile __global float *)(n + x1 * temp_w + y1), w1);
    AtomicAdd((volatile __global float *)(n + x1 * temp_w + (y1 + 1)), w2);
    AtomicAdd((volatile __global float *)(n + (x1 + 1) * temp_w + y1), w3);
    AtomicAdd((volatile __global float *)(n + (x1 + 1) * temp_w + (y1 + 1)), w4);

    return;
}