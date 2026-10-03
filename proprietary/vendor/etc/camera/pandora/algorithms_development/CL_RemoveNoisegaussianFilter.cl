// CL Kernel Source Code for RemoveNoisegaussianFilter Algorithm
// Version 1.0.0

__kernel void RemoveNoisegaussianFilter(
    __global float* input,
    __global float* output,
    int id)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    int index = width * height * id + y *width + x;

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
            sum += input[py * width + px] * w;
        }
    }
    output[index] = ((float)sum / 1019);
};
