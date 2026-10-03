// CL Kernel Source Code for DeNoisegaussianFilter5x5 Algorithm
// Version 1.0.0

__kernel void DeNoisegaussianFilter5x5(
    __global float* input,
    __global float* output,
    int id)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    int index = width * height * id + y *width + x;

    constant int gaussian_kernel[5][5] = {
        { 5, 17, 26, 17, 5},
        { 17, 61, 92, 61, 17},
        { 26,  92, 139, 92, 26},
        { 17, 61, 92, 61, 17},
        { 5, 17, 26, 17, 5},
    };

    if (x >= width || y >= height) {
        return;
    }

    float sum = 0;
    for (int j = -2; j <= 2; j++) {
        for (int i = -2; i <= 2; i++) {
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
            const int w = gaussian_kernel[j + 2][i + 2];
            sum += input[py * width + px] * w;
        }
    }
    output[index] = ((float)sum / 1011);
};
