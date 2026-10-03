// CL Kernel Source Code for ColorSketchMerge1 Algorithm
// Version 1.0.0

#define MIN_ITEM(A,B) (A>B? B:A)
#define MIN_VALUE(A,B,C) MIN_ITEM(MIN_ITEM(A,B),C)

__kernel void ColorSketchMerge1(
    __global float *pRGB,
    __global float *pInvRGB,
    __global int *hist,
    __private const int datalength)
{
    int j = get_global_id(0); //width
    int i = get_global_id(1); //height
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int indexR = i * width + j;
    int indexG = indexR + datalength;
    int indexB = indexG + datalength;

    float aB = pRGB[indexB] * pInvRGB[indexB];
    float aG = pRGB[indexG] * pInvRGB[indexG];
    float aR = pRGB[indexR] * pInvRGB[indexR];

    float bB = aB * 0.8f;
    float bG = aG * 0.8f;
    float bR = aR * 0.8f;

    float cB = 255 - pInvRGB[indexB];
    float cG = 255 - pInvRGB[indexG];
    float cR = 255 - pInvRGB[indexR];
    cB = (cB <= 0) ? 1 : cB;
    cG = (cG <= 0) ? 1 : cG;
    cR = (cR <= 0) ? 1 : cR;

    float dB = bB / cB;
    float dG = bG / cG;
    float dR = bR / cR;
    dB = (dB < 0) ? 0 : (dB > 255) ? 255 : dB;
    dG = (dG < 0) ? 0 : (dG > 255) ? 255 : dG;
    dR = (dR < 0) ? 0 : (dR > 255) ? 255 : dR;

    float eB = pRGB[indexB] + dB;
    float eG = pRGB[indexG] + dG;
    float eR = pRGB[indexR] + dR;
    eB = (eB > 255) ? 255 : eB;
    eG = (eG > 255) ? 255 : eG;
    eR = (eR > 255) ? 255 : eR;

    pRGB[indexR] = eR;
    pRGB[indexG] = eG;
    pRGB[indexB] = eB;
    int val = (int)(MIN_VALUE(eB, eG ,eR));
    if (hist[val] == 0) {
        atomic_inc(&hist[val]);
    }
}

