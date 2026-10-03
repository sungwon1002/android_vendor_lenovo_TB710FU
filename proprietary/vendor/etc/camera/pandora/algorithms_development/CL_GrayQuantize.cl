// CL Kernel Source Code for GrayQuantize Algorithm
// Version 1.0.0

__kernel void GrayQuantize(
    __global float *pInputRGB,
    __global unsigned char *pOutputGRAY,
    __private const int datalength)
{
    int i = get_global_id(0); //width
    int j = get_global_id(1); // height
    int width = get_global_size(0);
    int height = get_global_size(1);
    int indexR = j * width + i;
    int indexG = indexR + datalength;
    int indexB = indexG + datalength;

    float gray = pInputRGB[indexR] * 0.299f +
        pInputRGB[indexG] * 0.587f +
        pInputRGB[indexB] * 0.114f;
    pOutputGRAY[indexR] = (uint)(gray / 32);
}
