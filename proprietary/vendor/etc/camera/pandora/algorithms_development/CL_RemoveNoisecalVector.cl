// CL Kernel Source Code for RemoveNoisecalVector Algorithm
// Version 1.0.0

__kernel void RemoveNoisecalVector(
    __global float* input,
    __global float* output,
    float max_aM,
    float min_aM,
    int id)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    int index = width * height * id + j * width + i;
    float  dynamic_aM = 10 * log10(max_aM / min_aM);
    float th = 0.15f * dynamic_aM;
    float temp = -(input[index] - th) * input[index] / th;
    input[index] = 4 + fabs(input[index] - th);
    output[index] = pow(*(input + index), temp);
};
