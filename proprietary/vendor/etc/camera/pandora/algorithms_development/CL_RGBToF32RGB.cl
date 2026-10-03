// CL Kernel Source Code for RGBToF32RGB Algorithm
// Version 1.0.0

__kernel void RGBToF32RGB(
    __global unsigned char *pIn,
    __global float *pOut)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int frameSize = width * height;
    int planeOffset = j * width + i;

    float r = pIn[planeOffset];
    float g = pIn[frameSize + planeOffset];
    float b = pIn[frameSize * 2 + planeOffset];

    pOut[planeOffset] = r;
    pOut[frameSize + planeOffset] = g;
    pOut[frameSize * 2 + planeOffset] = b;
}
