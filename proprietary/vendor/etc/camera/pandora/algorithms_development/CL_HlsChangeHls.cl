// CL Kernel Source Code for HlsChangeHls Algorithm
// Version 1.0.0

__kernel void HlsChangeHls(
    __global float *hls,
    __global float *hlsDest,
    __global float *hue1,
    __global float *hue2,
    __global char *ind,
    __global float *ind_blur,
    int offset,
    int offset1,
    int offset2)
{
   int j = get_global_id(0);
   int i = get_global_id(1);
   int width  = get_global_size(0);
   int height = get_global_size(1);
   int frameSize = width * height;
   __global float *hue   = hls;
   __global float *light = hls + frameSize;
   __global float *sat   = hls + frameSize*2;
   __global float *outputH = hlsDest;
   __global float *outputL = hlsDest + frameSize;
   __global float *outputS = hlsDest + frameSize*2;

   float hue_value  = hue[i * width + j];
   float hue1_value = hue1[i * width + j];
   float hue2_value = hue2[i * width + j];

   float H = (((100 - offset) * hue_value + offset * hue2_value) / 100);
   H = H < 360 ? H : H - 360;
   H = H > 0 ? H : H + 360;
   if (offset != 0) {
       outputH[i * width + j] = H;
   } else {
       outputH[i * width + j] = hue_value;
   }

   float sat_val = sat[i * width + j];
   float light_val = light[i * width + j];
   float sat1    = sat_val;
   float blurred_img_val =ind_blur[i * width + j];
   float ind_val = (float)ind[i * width + j];
   if (offset1 > 0) {
       float s_ratio = (float )offset1 / 100 * (1.0f - sat_val);
       sat1 = sat_val * (1 + s_ratio * blurred_img_val);
       outputS[i * width + j] = sat1 > 1 ? 1: sat1;

   } else if (offset1 < 0) {
       float s_ratio = (float )offset1 / 100 * 0.5f;
       sat1 = sat_val * (1 + s_ratio * blurred_img_val);
       outputS[i * width + j] = sat1 > 1 ? 1: sat1;
   } else {
       outputS[i * width + j] = sat_val;
   }
   if (offset2 > 0) {
       float l_ratio = (float )offset2 / 100.0f;
       float light1 = light_val * (1 + l_ratio * (1 - light_val) *
           0.4f * sat_val * blurred_img_val);
       outputL[i * width + j] = light1 > 1 ? 1 : light1;
       if(light_val >= 0.5f) {
           sat1 = sat1 * l_ratio * 0.1f;
       } else {
           sat1 = -1 * sat1 * l_ratio * 0.1f;
       }
       sat1 = sat_val * (1 + sat1 * ind_val);
       outputS[i * width + j] = sat1 > 1 ? 1 : sat1;
   } else {
       outputL[i * width + j] = light_val;
   }
}

