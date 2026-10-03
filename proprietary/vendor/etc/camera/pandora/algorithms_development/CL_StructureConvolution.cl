// CL Kernel Source Code for StructureConvolution Algorithm
// Version 1.0.0

__kernel void StructureConvolution(
    __global float *input,
    __global float *output,
    int stride)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    constant int thick_kernel[] =
        { 0,  -2,  -9,  -15, -9,  -2,   0,
         -2,  -22, -67, -89, -67, -22, -2,
         -9,  -67, -46, 144, -46, -67, -9,
         -15, -89, 144, 736, 144, -89,-15,
         -9,  -67, -46, 144, -46, -67, -9,
         -2,  -22, -67, -89, -67, -22, -2,
          0,  -2,  -9,  -15, -9,  -2,   0 };

    int kernel_size = 7;
    float sum = 0;

    for (int i = -kernel_size / 2; i <= kernel_size / 2; i++) {
        for (int j = -kernel_size / 2; j <= kernel_size / 2; j++) {
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
            sum += thick_kernel[(i + kernel_size / 2) * kernel_size + j + kernel_size / 2] * input[py * stride + px];
        }
    }
    output[y * stride + x] = sum / 656;
}