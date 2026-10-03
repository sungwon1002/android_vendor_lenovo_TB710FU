// CL Kernel Source Code for StructureWeight Algorithm
// Version 1.0.0

__kernel void StructureWeight(
    __global float *input,
    __global float *output,
    int stride)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    constant int threshold[] = {5, 8, 160, 300, -5, -8, -160, -300};
    constant float weight[]  = {1.2f, 1.5f, 1.0f, 1.0f, 1.2f, 1.5f, 1.0f, 1.0f};
    float finalWeight = 0.0f;
    float ratio[6];
    ratio[0] = (weight[1] - weight[0]) / (threshold[1] - threshold[0]);
    ratio[4] = (weight[2] - weight[1]) / (threshold[2] - threshold[1]);
    ratio[1] = (weight[3] - weight[2]) / (threshold[3] - threshold[2]);
    float a = threshold[5] - threshold[4];
    float b = threshold[6] - threshold[5];
    float c = threshold[7] - threshold[6];
    a = a < 0 ? -a : a;
    b = b < 0 ? -b : b;
    c = c < 0 ? -c : c;
    ratio[2] = (weight[5] - weight[4]) / a;
    ratio[5] = (weight[6] - weight[5]) / b;
    ratio[3] = (weight[7] - weight[6]) / c;
    const int index = j * stride + i;
    if (input[index] < threshold[0] && input[index] >= 0) {
        finalWeight = weight[0];
    } else if (input[index] >= threshold[0] && input[index] < threshold[1]) {
        finalWeight = (input[index] - threshold[0]) * ratio[0] + weight[0];
    } else if (input[index] >= threshold[1] && input[index] < threshold[2]) {
        finalWeight = (input[index] - threshold[1]) * ratio[4] + weight[1];
    } else if (input[index] >= threshold[2]) {
        finalWeight = weight[2] + (input[index] - threshold[2]) * ratio[1];
    } else if (input[index] > threshold[4] && input[index] <= 0) {
        finalWeight = weight[4];
    } else if (input[index] <= threshold[4] && input[index] > threshold[5]) {
        float abs_convolution = input[index] - (threshold[4]);
        abs_convolution = abs_convolution>=0.0f?abs_convolution : -abs_convolution;
        finalWeight = abs_convolution * ratio[2] + weight[4];
    } else if (input[index] <= threshold[5] && input[index] > threshold[6]) {
        float abs_convolution = input[index] - (threshold[5]);
        abs_convolution = abs_convolution>=0.0f?abs_convolution : -abs_convolution;
        finalWeight = abs_convolution * ratio[5] + weight[5];
    } else if (input[index] <= threshold[6]) {
        float abs_convolution = input[index] - (threshold[6]);
        abs_convolution = abs_convolution>=0.0f?abs_convolution : -abs_convolution;
        finalWeight = weight[6] + abs_convolution * ratio[3];
    }
    output[index] = (input[index] * finalWeight);
}
