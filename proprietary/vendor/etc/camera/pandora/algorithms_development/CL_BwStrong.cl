// CL Kernel Source Code for BwStrong Algorithm
// Version 1.0.0

#define FORMAT_NV21   0 // YVU
#define FORMAT_NV12   1 // YUV
#define FORMAT_YUV_3P 2
#define FORMAT_RGB    9
#define FORMAT_HLS    10
#define FORMAT_F32RGB 11
#define FORMAT_F32HLS 12

#define MIN_ITEM(A,B) (A>B? B:A)
#define MIN_VALUE(A,B,C) MIN_ITEM(MIN_ITEM(A,B),C)
#define MAX_ITEM(A,B) (A>B? A:B)
#define MAX_VALUE(A,B,C) MAX_ITEM(MAX_ITEM(A,B),C)

void BwStrongRGB(
    __global uchar *pInput,
    __global uchar *pOutput,
    float k,
    int enable_lut,
    __global float *pLut)
{

    int c1 = 4;
    int c2 = 4;
    int c3 = 2;
    int c4 = 6;
    int c5 = 6;
    int c6 = 8;
    int i = get_global_id(0);
    int j = get_global_id(1);
    int dim = get_global_id(2);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int datalength = width * height;
    float bvalue = pInput[i + j * width + 2 * datalength];
    float gvalue = pInput[i + j * width + 1 * datalength];
    float rvalue = pInput[i + j * width + 0 * datalength];
    float min_val = MIN_VALUE(rvalue,gvalue,bvalue);
    float max_val = MAX_VALUE(rvalue,gvalue,bvalue);
    float sum_value = rvalue + gvalue + bvalue;
    float mid_val = sum_value - max_val - min_val;
    float ratio_max_mid;
    float ratio_max;
    if (max_val == rvalue) {
        ratio_max = c1;
    } else if (max_val == gvalue) {
        ratio_max = c2;
    } else {
        max_val = bvalue;
        ratio_max = c3;
    }
    if (rvalue<=gvalue && rvalue<=bvalue) { // g+ b = cyan
        min_val = rvalue;
        ratio_max_mid = c5;
    } else if (bvalue <= rvalue && bvalue<=gvalue) { //r+g = yellow
        min_val = bvalue;
        ratio_max_mid = c4;
    } else { //r+b = m
        min_val = gvalue;
        ratio_max_mid = c6;
    }
    float y = ((max_val - mid_val) * ratio_max +
        (mid_val - min_val) * ratio_max_mid)/10 + min_val;
    if(enable_lut) {
        y =  y * (1.0f - 0.8f) + pLut[(int)y] * 0.8f;
    }
    float out = rvalue * (1 - k) + y * k;
    pOutput[i + j * width + 0 * datalength] = out;
    out = gvalue * (1 - k) + y * k;
    pOutput[i + j * width + 1 * datalength] = out;
    out = bvalue * (1 - k) + y * k;
    pOutput[i + j * width + 2 * datalength] = out;
}

void BwStrongF32RGB(
    __global float *pInput,
    __global float *pOutput,
    float k,
    int enable_lut,
    __global float *pLut)
{

    int c1 = 4;
    int c2 = 4;
    int c3 = 2;
    int c4 = 6;
    int c5 = 6;
    int c6 = 8;
    int i = get_global_id(0);
    int j = get_global_id(1);
    int dim = get_global_id(2);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int datalength = width * height;
    float bvalue = pInput[i + j * width + 2 * datalength];
    float gvalue = pInput[i + j * width + 1 * datalength];
    float rvalue = pInput[i + j * width + 0 * datalength];
    float min_val = MIN_VALUE(rvalue,gvalue,bvalue);
    float max_val = MAX_VALUE(rvalue,gvalue,bvalue);
    float sum_value = rvalue + gvalue + bvalue;
    float mid_val = sum_value - max_val - min_val;
    float ratio_max_mid;
    float ratio_max;
    if (max_val == rvalue) {
        ratio_max = c1;
    } else if (max_val == gvalue) {
        ratio_max = c2;
    } else {
        max_val = bvalue;
        ratio_max = c3;
    }
    if (rvalue<=gvalue && rvalue<=bvalue) { // g+ b = cyan
        min_val = rvalue;
        ratio_max_mid = c5;
    } else if (bvalue <= rvalue && bvalue<=gvalue) { //r+g = yellow
        min_val = bvalue;
        ratio_max_mid = c4;
    } else { //r+b = m
        min_val = gvalue;
        ratio_max_mid = c6;
    }
    float y = ((max_val - mid_val) * ratio_max +
        (mid_val - min_val) * ratio_max_mid)/10 + min_val;
    if(enable_lut) {
        y =  y * (1.0f - 0.8f) + pLut[(int)y] * 0.8f;
    }
    float out = rvalue * (1 - k) + y * k;
    pOutput[i + j * width + 0 * datalength] = out;
    out = gvalue * (1 - k) + y * k;
    pOutput[i + j * width + 1 * datalength] = out;
    out = bvalue * (1 - k) + y * k;
    pOutput[i + j * width + 2 * datalength] = out;
}


__kernel void BwStrong(
    __global void *pInput,
    int format,
    __global void *pOutput,
    float k,
    int enable_lut,
    __global float *pLut)
{
    if(format == FORMAT_RGB){
        BwStrongRGB(pInput, pOutput, k, enable_lut, pLut);
    } else {
        BwStrongF32RGB(pInput, pOutput, k, enable_lut, pLut);
    }
}
