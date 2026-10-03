__kernel void nv21ToYUV(
    __global unsigned char *pIn,
    __global unsigned char *pOut,
    int width,
    int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    pOut[y * width + x] = pIn[y * width + x];
    if (x % 2 == 0 && y % 2 == 0) {
        int planeOffset = (y / 2) * (width / 2) + x / 2;
        int uvOffset    = width * height + y / 2 * width + x;
        pOut[width * height + planeOffset] = pIn[uvOffset + 1];
        pOut[width * height * 5 / 4 + planeOffset] = pIn[uvOffset];
    }
}

