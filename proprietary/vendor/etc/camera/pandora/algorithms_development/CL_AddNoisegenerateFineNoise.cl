// CL Kernel Source Code for generateFineNoise Algorithm
// Version 1.0.0

__kernel void AddNoisegenerateFineNoise(
    __global float *inputY,
   float mean,
   float std,
   float density)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int index = y * width + x;
    int min_value = (int)(2 * mean - sqrt(12 * std));
    int max_value = (int)(2 * mean + sqrt(12 * std));

    if (( density * x * y / (x + y + 1 ) * x + density * y * x /(x + y +1)) / (x + 1)/ (y +1 )  < density) {
        int value =(((x + 1) + height + (y +1) + width ) * height * width ) % (max_value - min_value) + min_value;
        inputY[index] = value;
    } else {
        inputY[index] = 0;
    }
}
