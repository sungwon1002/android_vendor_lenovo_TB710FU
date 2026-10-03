// CL Kernel Source Code for StructureGaussianBlur Algorithm
// Version 1.0.0

__kernel void StructureGaussianBlur(
    __global float *input,
    __global float *output,
    int stride,
    int scanline,
    int isNV21)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);

    if (x >= width || y >= height) {
        return;
    }

    float sum = 0;

    constant int gaussian_kernel[3][3] = {
        { 58, 127, 58},
        { 127, 279, 127},
        { 58, 127, 58},
    };

    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int px = x + i;
            int py = y + j;
            if (px < 0) {
               px = -px;
            } else if (px > width - 1) {
               px = width + width - px - 2;
            }
            if (py < 0) {
               py = -py;
            } else if (py > height - 1) {
               py = height + height - py - 2;
            }
            const int w = gaussian_kernel[j + 1][i + 1];
            sum += input[py * stride + px] * w;
        }
    }
    output[y * stride + x] = sum / 1019;
    int frameSize = stride * scanline;
    if (isNV21 == 1) {
        int planeOffset = y * stride + x;
        int nvIndex = y / 2 * stride + x - x % 2;
        output[frameSize + nvIndex + 1] = input[frameSize + nvIndex + 1];
        output[frameSize + nvIndex] = input[frameSize + nvIndex];
    } else {
        int uvOffset = (y / 2) * (width / 2) + x / 2;
        int uIndex = frameSize + uvOffset;
        int vIndex = uIndex + frameSize / 4;
        output[uIndex] = input[uIndex];
        output[vIndex] = input[vIndex];
    }
}