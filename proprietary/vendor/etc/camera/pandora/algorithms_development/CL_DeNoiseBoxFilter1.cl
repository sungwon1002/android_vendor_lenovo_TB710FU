// CL Kernel Source Code for DeNoiseBoxFilter1 Algorithm
// Version 1.0.0

__kernel void DeNoiseBoxFilter1(
    __global float *input_image,
    __global float *output_image,
    int kernel_size, int id)
{
    int width = get_global_size(0);
    int height = get_global_size(1);
    int x = get_global_id(0);
    int y = get_global_id(1);
    int radius = (kernel_size / 2);
    int col_beg  = max(x - radius, 0);
    int col_end = min(x + radius, width -1);
    int base = width * height * id;
    float sum = 0;
    int count = 0;
    for (int i = col_beg ; i <= col_end; i++) {
       sum += input_image[y * width + i + base];
       ++count;
    }
    output_image[y * width + x + base] = sum / count;
}
