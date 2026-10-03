// CL Kernel Source Code for DCCubic Algorithm
// Version 1.0.0

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

__kernel void DCCubicProcessYFunc(
    __global uchar *pInput,
    __global float *preData,
    __global uchar *pOutput,
    int stride, int scanline)
{
    int j = get_global_id(0);
    int i = get_global_id(1);

    int width  = get_global_size(0);
    int height = get_global_size(1);
    int datalength = width * height;

    float luma1;
    int   x1, y1, x2, y2;
    float x,  y;
    constant float w1[11] = {0,     -0.081f, -0.128f, -0.147f, -0.144f, -0.125f,
                            -0.096f, -0.063f,  0.032f, -0.009f,  0};
    constant float w2[11] = {1.0f,      0.981f,  0.928f,  0.847f,  0.744f,  0.625f,
                             0.496f,  0.363f,  0.232f,  0.109f,  0};
    constant float w3[11] = {0,      0.1089f, 0.2319f, 0.363f,  0.496f, 0.625f,
                             0.744f,  0.847f,  0.928f,  0.981f,  1};
    constant float w4[11] = {0,     -0.0089f, -0.0319f, -0.063f, -0.096f, -0.125f,
                            -0.144f, -0.147f,  -0.128f,  -0.081f,  0};

    x = preData[j + i * width];
    y = preData[j + i * width + datalength];

    x = fmin(fmax(1.0f, x), height - 3.0f);
    y = fmin(fmax(1.0f, y), width - 3.0f);

    x1 = (int)x;
    y1 = (int)y;
    x2 = (int)((x - (int)(x)) * 10);
    y2 = (int)((y - (int)(y)) * 10);

    float W =
        w1[x2] * w2[y2] + w1[x2] * w3[y2] + w2[x2] * w1[y2] +
        w2[x2] * w2[y2] + w2[x2] * w3[y2] + w2[x2] * w4[y2] +
        w3[x2] * w1[y2] + w3[x2] * w2[y2] + w3[x2] * w3[y2] +
        w3[x2] * w4[y2] + w4[x2] * w2[y2] + w4[x2] * w3[y2];

    luma1 =
        pInput[(x1 - 1) * stride + y1] * w1[x2] * w2[y2] +
        pInput[(x1 - 1) * stride + y1 + 1] * w1[x2] * w3[y2] +
        pInput[x1 * stride + y1 - 1 ] * w2[x2] * w1[y2] +
        pInput[x1 * stride + y1 ] * w2[x2] * w2[y2] +
        pInput[x1 * stride + y1 + 1 ] * w2[x2] * w3[y2] +
        pInput[x1 * stride + y1 + 2 ] * w2[x2] * w4[y2] +
        pInput[(x1 + 1) * stride + y1 - 1 ] * w3[x2] * w1[y2] +
        pInput[(x1 + 1) * stride + y1 ] * w3[x2] * w2[y2] +
        pInput[(x1 + 1) * stride + y1 + 1 ] * w3[x2] * w3[y2] +
        pInput[(x1 + 1) * stride + y1 + 2 ] * w3[x2] * w4[y2] +
        pInput[(x1 + 2) * stride + y1 ] * w4[x2] * w2[y2] +
        pInput[(x1 + 2) * stride + y1 + 1 ] * w4[x2] * w3[y2];

    pOutput[i * stride + j ] = MIN(MAX((luma1 / W), 0), 255);
}