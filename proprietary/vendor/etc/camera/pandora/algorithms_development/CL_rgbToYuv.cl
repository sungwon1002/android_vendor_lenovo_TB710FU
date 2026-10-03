// CL Kernel Source Code for rgbToYuv Algorithm
// Version 1.0.0

__kernel void rgbToYuv(
    __global unsigned char *pIn,
    __global unsigned char *pOut)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int frameSize   = width * height;
    int planeoffset = width * j + i;
    int uvOffset = (j / 2) * (width / 2) + i / 2;
    int uIndex = frameSize + uvOffset;
    int vIndex = uIndex + frameSize / 4;

    int r = pIn[planeoffset];
    int g = pIn[planeoffset + frameSize];
    int b = pIn[planeoffset + frameSize * 2];

    int y = 0.229f * r + 0.587f * g + 0.114f * b;
    int u = -0.1684f * r - 0.3316f * g + 0.5f * b + 128;
    int v = 0.5f * r - 0.4187f * g - 0.0813f * b + 128;

    if (y > 255) y = 255;
    if (u > 255) u = 255;
    if (v > 255) v = 255;
    if (y < 0) y = 0;
    if (u < 0) u = 0;
    if (v < 0) v = 0;

    pOut[planeoffset] = y;
    if (i % 2 == 0 && j % 2 == 0) {
        pOut[uIndex] = u;
        pOut[vIndex] = v;
    }
}
