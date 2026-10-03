// CL Kernel Source Code for Sharpness410GaussianFilter Algorithm
// Version 1.0.0

__kernel void Sharpness410GaussianFilter(
    __global const uchar *input,
    __global float *output,
    int stride,
    int scanline)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int index = y * width + x;

    constant int gaussian_kernel[3][3] = {
        { 58,  127, 58  },
        { 127, 279, 127 },
        { 58,  127, 58  },
    };

    if (x >= width || y >= height) {
        return;
    }

    float sum = 0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int px = x + i;
            int py = y + j;
            if (px < 0) {
                px = abs(px);
            } else if (px > width - 1) {
                px = width + width - px - 2;
            }
            if (py < 0) {
                py = abs(py);
            } else if (py > height - 1) {
                py = height + height - py - 2;
            }
            const int w = gaussian_kernel[j + 1][i + 1];
            sum += input[py * stride + px] * w;
        }
    }
    output[index] = ((float)sum / 1019);
}