// CL Kernel Source Code for SkinChangeLS Algorithm
// Version 1.0.0

__kernel void SkinChangeLS(
    __global float *hls,
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
    __global float *light = hls + frameSize;
    __global float *sat   = hls + frameSize * 2;
    __global float *outputL = light;
    __global float *outputS = sat;
    float sat_val   = sat[i * width + j];
    float light_val = light[i * width + j];
    float sat1;
    float blurred_img_val =ind_blur[i * width + j];

    if (offset1 > 0) {
        float s_ratio = (float )offset1 / 100 * (1.0f - sat_val);
        sat1 = sat_val * (1 + s_ratio * blurred_img_val);
        outputS[i * width + j] = sat1 > 1 ? 1 : sat1;
    } else if (offset1 < 0) {
       float s_ratio = (float )offset1 / 100 * 0.3f;
       sat1 = sat_val * (1 + s_ratio * blurred_img_val);
       outputS[i * width + j] = sat1 > 1 ? 1 : sat1;
    }

    if (offset2 > 0) {
       float l_ratio = (float )offset2 / 100.0f;
       float light1 = light_val * (1 + l_ratio * (1 - light_val) * 0.6f * sat_val * blurred_img_val);
       outputL[i * width + j] = light1 > 1 ? 1: light1;
    } else if (offset2 < 0) {
       float l_ratio = (float )offset2 / 100.0f;
       float light1 = light_val * (1 + l_ratio * (1 - light_val) * sat_val * blurred_img_val);
       outputL[i * width + j] = light1 > 1 ? 1: light1;
    }
}

