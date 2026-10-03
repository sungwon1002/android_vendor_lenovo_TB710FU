// CL Kernel Source Code for Sharpness410Convolution Algorithm
// Version 1.0.0

__kernel void Sharpness410Convolution(
    __global float *input_image,
    __global float *output_image,
    __global int *kernel_conv,
    int kernel_size)
{
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int x = get_global_id(0);
    int y = get_global_id(1);
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
                input_image[py * width + px];
        }
    }
    output_image[y * width + x] = sum / 656;
}
