// CL Kernel Source Code for DefoggingMaxminLimit Algorithm
// Version 1.0.0

__kernel void DefoggingMaxMinLimit(
    __global float* input,
    float min,
    float max)
{
    int index = get_global_id(0);
    float cache = input[index];
    cache = fmin(fmax(cache, min), max);
    input[index] = cache;
};