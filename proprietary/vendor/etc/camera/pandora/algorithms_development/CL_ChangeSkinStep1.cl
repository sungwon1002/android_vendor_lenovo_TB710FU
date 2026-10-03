// CL Kernel Source Code for ChangeSkinStep1 Algorithm
// Version 1.0.0

__kernel void ChangeSkinStep1(
    __global float *pIn,
    __global float *pOut,
    __global float *pHue,
    __global char *pInd,
    __global char *pLut,
    int8 colorInfo,
    int offset)
 {
    int j = get_global_id(0);
    int i = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int frameSize = width * height;

    #define MAX(a, b)   ((a) > (b) ? (a) : (b))
    #define MIN(a, b)   ((a) < (b) ? (a) : (b))
    #define CLIP(x,a,b) (MIN(MAX(x,a),b))
    #define ABS(x)      ((x) > (0) ? (x) : -1 * (x))

    __global float *inputR = pIn;
    __global float *inputG = inputR + frameSize;
    __global float *inputB = inputG + frameSize;
    __global float *outputR = pOut;
    __global float *outputG = outputR + frameSize;
    __global float *outputB = outputG + frameSize;

    int   center  = colorInfo.s0;
    int   range1  = colorInfo.s1;
    int   low     = colorInfo.s2;
    int   high    = colorInfo.s3;
    float h_ratio = (float)colorInfo.s4;
    int   pos     = i * width + j;
    float h_value = pHue[pos];

    float b = inputB[pos];
    float g = inputG[pos];
    float r = inputR[pos];

    /* Right pixels*/
    if ((h_value >= 0 && h_value < high) ||
        (h_value  > low && h_value <= 360)) {
        b = b * (1 - h_ratio) + pLut[2 * 256 + (int)b] * h_ratio;
        g = g * (1 - h_ratio) + pLut[1 * 256 + (int)g] * h_ratio;
        r = r * (1 - h_ratio) + pLut[0 * 256 + (int)r] * h_ratio;
        outputB[pos] = CLIP(b, 0, 255);
        outputG[pos] = CLIP(g, 0, 255);
        outputR[pos] = CLIP(r, 0, 255);
        pInd[i * width + j] = 1;
    } else {
       outputB[pos] = CLIP(b, 0, 255);
       outputG[pos] = CLIP(g, 0, 255);
       outputR[pos] = CLIP(r, 0, 255);
       pInd[pos] = 0;
    }
}
