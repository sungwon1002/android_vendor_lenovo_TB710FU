// CL Kernel Source Code for DefoggingBoxFilter Algorithm
// Version 1.0.0

__kernel void DefoggingBoxFilter(
    __global float *input_image,
    __global float *output_image,
    int kernel_size)
{
    int width = get_global_size(0);
    int height = get_global_size(1);
    int x = get_global_id(0);
    int y = get_global_id(1);
    int a = (kernel_size % 2 == 0);
    float sum = 0;
    for (int i = -kernel_size / 2; i <= kernel_size / 2; i++) {
      for (int j = -kernel_size / 2; j <= kernel_size / 2; j++) {
          const int px = clamp(x + i, 0, width - 1);
          const int py = clamp(y + j, 0, height - 1);
          sum += input_image[py * width + px];
      }
    }
    output_image[y * width + x] = sum / (kernel_size + a) / (kernel_size + a);
}