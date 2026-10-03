// CL Kernel Source Code for Sharpness410Weight Algorithm
// Version 1.0.0

__kernel void Sharpness410Weight(
    __global float *img_convolution,
    __global float *sharp_img,
    __global int *threshold,
    __global float *weight,
    __global float *ratio)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int index = y * width + x;
    float final_weight = 0.0f;

    if (img_convolution[index] < threshold[0] && img_convolution[index] >= 0) {
        final_weight = weight[0];
    } else if (img_convolution[index] >= threshold[0] &&
        img_convolution[index] < threshold[1]) {
        final_weight = (img_convolution[index] - threshold[0]) * ratio[0] + weight[0];
    } else if (img_convolution[index] >= threshold[1] &&
        img_convolution[index] < threshold[2]) {
        final_weight = (img_convolution[index] - threshold[1]) * ratio[4] + weight[1];
    } else if (img_convolution[index] >= threshold[2]) {
        final_weight = weight[2] + (img_convolution[index] - threshold[2]) * ratio[1];
    } else if (img_convolution[index] > threshold[4] &&
        img_convolution[index] <= 0) {
        final_weight = weight[4];
    } else if (img_convolution[index] <= threshold[4] &&
        img_convolution[index] > threshold[5]) {
        float abs_convolution = img_convolution[index] - (threshold[4]);
        abs_convolution = abs_convolution >= 0.0f ? abs_convolution : -abs_convolution;
        final_weight = abs_convolution * ratio[2] + weight[4];
    } else if (img_convolution[index] <= threshold[5] &&
        img_convolution[index] > threshold[6]) {
        float abs_convolution = img_convolution[index] - (threshold[5]);
        abs_convolution = abs_convolution >= 0.0f ? abs_convolution : -abs_convolution;
        final_weight = abs_convolution * ratio[5] + weight[5];
    } else if (img_convolution[index] <= threshold[6]) {
        float abs_convolution = img_convolution[index] - (threshold[6]);
        abs_convolution = abs_convolution >= 0.0f ? abs_convolution : -abs_convolution;
        final_weight = weight[6] + abs_convolution * ratio[3];
    }

    sharp_img[index] = (img_convolution[index] * final_weight);
}