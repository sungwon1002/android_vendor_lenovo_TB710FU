__kernel void yuvToNV21(
    __global unsigned char *pIn,
    __global unsigned char *pOut,
        int width,
        int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    pOut[y * width + x] = pIn[y * width + x];
    if (x % 2 == 0 && y % 2 == 0) {
        int uvOffset    = width * height + (y / 2) * width + x;
        int planeOffset = (y / 2) * (width / 2) + x / 2;
        pOut[uvOffset + 1] = pIn[width * height + planeOffset];
        pOut[uvOffset]  = pIn[width * height * 5 / 4 + planeOffset];
    }
}

