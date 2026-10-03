// CL Kernel Source Code for nv21ToYuvRemoveStrideScanline Algorithm
// Version 1.0.0

__kernel void nv21ToYuvRemoveStrideScanline(
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

    pOut[i + widthJ] = pIn[i + strideJ];
    if (i % 2 == 0 && j % 2 == 0) {
        int planeOffset = i / 2 +  j / 2 * width / 2;
        int uvOffset    = sxc + i + j / 2 * stride;
        pOut[wxh + planeOffset] = pIn[uvOffset + 1];
        pOut[wxh5D4 + planeOffset] = pIn[uvOffset];
    }
}
