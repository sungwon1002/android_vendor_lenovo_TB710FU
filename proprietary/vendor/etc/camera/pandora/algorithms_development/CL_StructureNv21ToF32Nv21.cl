// CL Kernel Source Code for StructureNv21ToF32Nv21 Algorithm
// Version 1.0.0

__kernel void StructureNv21ToF32Nv21(
    __global unsigned char *pIn,
    __global float *pOut,
    int stride,
    int scanline)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int planeOffset = j * stride + i;
    int frameSize = stride * scanline;
    int nvIndex = j / 2 * stride + i - i % 2;

    pOut[planeOffset] = pIn[planeOffset];
    pOut[frameSize + nvIndex + 1] = pIn[frameSize + nvIndex + 1];
    pOut[frameSize + nvIndex] = pIn[frameSize + nvIndex];
}
