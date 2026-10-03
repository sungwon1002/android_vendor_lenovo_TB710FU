// CL Kernel Source Code for yuvToNv12AddStrideScanline Algorithm
// Version 1.0.0

__kernel void yuvToNv12AddStrideScanline(
    __global unsigned char *pIn,
    __global unsigned char *pOut,
    int width,
    int stride,
    int wxh,
    int wxh5D4,
    int sxc)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int widthJ  = width * j;
    int strideJ = stride * j;

    pOut[i + strideJ] = pIn[i + widthJ];
    if (i % 2 == 0 && j % 2 == 0) {
        int uvOffset    = i + sxc + strideJ / 2;
        int planeOffset = i / 2 + j / 2 * width / 2;
        pOut[uvOffset] = pIn[wxh + planeOffset];
        pOut[uvOffset + 1]  = pIn[wxh5D4 + planeOffset];
    }
}
