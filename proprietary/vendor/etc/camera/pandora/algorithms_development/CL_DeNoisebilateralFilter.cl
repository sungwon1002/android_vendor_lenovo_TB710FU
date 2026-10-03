// CL Kernel Source Code for DeNoisebilateralFilter Algorithm
// Version 1.0.0

__kernel void DeNoisebilateralFilter(
    __global float *pSrc,
    __global float *pDest,
    __global float *colorDistTablePtr,
    int radius,
    int id)
{
    int X = get_global_id(0);
    int Y = get_global_id(1);

    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int base = width * height * id;

    int Py = Y * width + base;
    __global float *LinePD = &pDest[Py];
    __global float *LinePS = &pSrc[Py];
    float sumPix = 0;
    float sum = 0;
    float factor = 0;
    for (int i = -radius; i <= radius; i++)
    {
        __global float  *pLine = &pSrc[base];
        int cPix = 0;
        for (int j = -radius; j <= radius; j++)
        {
            cPix = pLine[((Y + i + height) % height) * width +  (X + j + width) % width];
            int pos = (int)LinePS[X] * 256 +  cPix;
            factor = colorDistTablePtr[pos % (256 * 256)];
            sum += factor;
            sumPix += (factor * cPix);
        }
    }
    LinePD[X] = (sumPix / sum);
}
