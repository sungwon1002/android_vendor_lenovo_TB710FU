// CL Kernel Source Code for RemoveNoisefabsImage Algorithm
// Version 1.0.0

__kernel void RemoveNoisefabsImage(
    __global float* input,
    int id)
{
   int width = get_global_size(0);
   int height = get_global_size(1);
   int x = get_global_id(0);
   int y = get_global_id(1);
   int index = y * width + x + width * height * id;
   input[index] = fabs(input[index]);
}
