// CL Kernel Source Code for StructureSharpStrength Algorithm
// Version 1.0.0


__kernel void StructureSharpStrength(
    __global float *input,
    __global float *sharpImg,
    int stride)
{
   int i = get_global_id(0);
   int j = get_global_id(1);
   const int index = j * stride + i;

   constant float strength[] = {1.4f, 1.5f, 1.5f, 1.2f};
   constant int luma[] = { 20, 160, 200 };
   float scope[3];
   scope[0] = (strength[1] - strength[0]) / (luma[1] - luma[0]);
   scope[1] = (strength[2] - strength[1]) / (luma[2] - luma[1]);
   scope[2] = (strength[3] - strength[2]) / (255 - luma[2]);

   float yData = input[index];
   const float sharpY = sharpImg[index];
   float strengthVal = 0;

   if (yData < luma[0]) {
       strengthVal = sharpY * (strength[0]);
   } else if (yData >= luma[0] && yData < luma[1]) {
       strengthVal = sharpY * (strength[0] + scope[0] * (yData - luma[0]));
   } else if (yData >= luma[1] && yData < luma[2]) {
       strengthVal = sharpY * (strength[1] + scope[1] * (yData - luma[1]));
   } else if (yData >= luma[2]) {
       strengthVal = sharpY * (strength[2] + scope[2] * (yData - luma[2]));
   }
   yData = yData + strengthVal;
   yData = yData < 0 ? 0 : yData;
   input[index] = yData > 255 ? 255 : yData;
}
