// CL Kernel Source Code for DeNoiseedgeDetective Algorithm
// Version 1.0.0

__kernel void DeNoiseedgeDetective(
    __global float *input,
    __global float *output,
    __global float *img_convolution,
    __global float *filter_img,
    float max,
    float min,
    int id)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int base = width * height * id;

    if (img_convolution[i + j * width + base] >= 0)
        img_convolution[i + j * width + base] = img_convolution[i + j * width + base] / max;
    if (img_convolution[i + j * width + base] < 0)
        img_convolution[i + j * width + base] = img_convolution[i + j * width + base] / min;

    if (img_convolution[i + j * width + base] >= (-0.01f)  && (img_convolution[i + j * width + base] <= 0.01f))
        output[i + j * width + base] = filter_img[i + j * width + base];
    if ((img_convolution[i + j * width + base] < -0.01f) || (img_convolution[i + j * width + base] > 0.01f))
        output[i + j * width + base] = input[i + j * width + base];
}
