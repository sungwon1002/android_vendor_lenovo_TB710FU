// CL Kernel Source Code for DeNoiseclampImage Algorithm
// Version 1.0.0

__kernel void DeNoiseclampImage(
    __global float* input,
    int id)
{
   int width = get_global_size(0);
   int height = get_global_size(1);
   int x = get_global_id(0);
   int y = get_global_id(1);
   int index = y * width + x + width * height * id;

   float temp  = input[index];
   if (temp > 255) {
       temp = 255;
   } else if (temp < 0) {
       temp = 0;
   }
   input[index] = temp;
}

