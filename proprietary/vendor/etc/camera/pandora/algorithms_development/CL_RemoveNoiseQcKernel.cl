// CL Kernel Source Code for RemoveNoiseQcKernel Algorithm
// Version 1.0.0

constant int kernel_size = 7;
constant int kernel_conv[7 * 7] = {
         0, -2, -9, -15, -9, -2, 0,
        -2, -22, -67, -89, -67, -22, -2,
        -9, -67, -46, 144, -46, -67, -9,
        -15, -89, 144, 736, 144, -89, -15,
        -9, -67, -46, 144, -46, -67, -9,
        -2, -22, -67, -89, -67, -22, -2,
         0, -2, -9, -15, -9, -2, 0
};

__kernel void RemoveNoiseQcKernel(
    __global const float *input,
    __global float *output,
    int id)
{
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int datalength = width * height;
    int x = get_global_id(0);
    int y = get_global_id(1);
    int base = width * height * id;
    float sum = 0;

    for (int i = -kernel_size / 2; i <= kernel_size / 2; i++) {
        for (int j = -kernel_size / 2; j <= kernel_size / 2; j++) {
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
            sum += kernel_conv[(i + kernel_size / 2) *
                kernel_size + j + kernel_size / 2] *
                input[py * width + px + base];
        }
    }
    output[y * width + x + base] = sum / 656;
}
