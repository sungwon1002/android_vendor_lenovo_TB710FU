// CL Kernel Source Code for ColorSketchMerge2 Algorithm
// Version 1.0.0

__kernel void ColorSketchMerge2(
    __global float *pRGB,
    __private const int minVal,
    __private const int invMinVal,
    __private const int datalength)
{
    int j = get_global_id(0); //width
    int i = get_global_id(1); //height
    int width = get_global_size(0);
    int height = get_global_size(1);
    int indexR = i * width + j;
    int indexG = indexR + datalength;
    int indexB = indexG + datalength;

    float eB = pRGB[indexB];
    float eG = pRGB[indexG];
    float eR = pRGB[indexR];

    eB = 255 * (eB - minVal);
    eG = 255 * (eG - minVal);
    eR = 255 * (eR - minVal);

    float gB = eB / invMinVal;
    float gG = eG / invMinVal;
    float gR = eR / invMinVal;
    gB = (gB < 0) ? 0 : (gB > 255) ? 255 : gB;
    gG = (gG < 0) ? 0 : (gG > 255) ? 255 : gG;
    gR = (gR < 0) ? 0 : (gR > 255) ? 255 : gR;

    pRGB[indexB] = gB;
    pRGB[indexG] = gG;
    pRGB[indexR] = gR;
}