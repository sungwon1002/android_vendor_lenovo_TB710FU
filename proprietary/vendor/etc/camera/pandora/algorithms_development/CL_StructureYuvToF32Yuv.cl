// CL Kernel Source Code for StructureYuvToF32Yuv Algorithm
// Version 1.0.0

__kernel void StructureYuvToF32Yuv(
    __global unsigned char *pIn,
    __global float *pOut)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int planeOffset = j * width + i;
    int frameSize = width * height;
    int uvOffset = (j / 2) * (width / 2) + i / 2;
    int uIndex = frameSize + uvOffset;
    int vIndex = uIndex + frameSize / 4;

    pOut[planeOffset] = pIn[planeOffset];
    pOut[uIndex] = pIn[uIndex];
    pOut[vIndex] = pIn[vIndex];
}
