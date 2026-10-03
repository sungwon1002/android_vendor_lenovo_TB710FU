// CL Kernel Source Code for StructureF32YuvToYuv Algorithm
// Version 1.0.0

__kernel void StructureF32YuvToYuv(
    __global float *pIn,
    __global unsigned char *pOut)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int planeOffset = j * width + i;
    int frameSize = width * height;
    int uvOffset  = (j / 2) * (width / 2) + i / 2;
    int uIndex = frameSize + uvOffset;
    int vIndex = uIndex + frameSize / 4;

    int y = pIn[planeOffset];
    int u = pIn[uIndex];
    int v = pIn[vIndex];

    if (y > 255) y = 255;
    if (u > 255) u = 255;
    if (v > 255) v = 255;
    if (y < 0) y = 0;
    if (u < 0) u = 0;
    if (v < 0) v = 0;

    pOut[planeOffset] = y;
    if (i % 2 == 0 && j % 2 == 0) {
        pOut[uIndex] = u;
        pOut[vIndex] = v;
    }
}