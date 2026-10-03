// CL Kernel Source Code for Vibrance Algorithm
// Version 1.0.0

#define FORMAT_RGB    9
#define MAX(a, b)   ((a) > (b) ? (a) : (b))
#define MIN(a, b)   ((a) < (b) ? (a) : (b))
#define CLIP(x,a,b) (MIN(MAX(x,a),b))
#define MAX_VALUE(A,B,C) MAX(MAX(A,B),C)
#define MIN_VALUE(A,B,C) MIN(MIN(A,B),C)

 void VibranceF32RGB(
    __global float *pInput,
    __global float *pOutput,
    float  increment)
{
    int i   = get_global_id(0);
    int j   = get_global_id(1);
    int dim = get_global_id(2);
    int width    = get_global_size(0);
    int height   = get_global_size(1);
    int datalength = width * height;
    float bvalue   = pInput[i + j * width + 2 * datalength];
    float gvalue   = pInput[i + j * width + 1 * datalength];
    float rvalue   = pInput[i + j * width + 0 * datalength];
    float min_val  = MIN_VALUE(rvalue,gvalue,bvalue);
    float max_val  = MAX_VALUE(rvalue,gvalue,bvalue);
    float saturation = max_val - min_val;
    float k = 1.0f + increment * (1.0f - saturation / 255.0f);
    float Y = 0.299f * rvalue + 0.587f * gvalue + 0.114f * bvalue;
    float Rc = Y * (1 - k) + rvalue * k;
    float Gc = Y * (1 - k) + gvalue * k;
    float Bc = Y * (1 - k) + bvalue * k;

    pOutput[i + j * width + 0 * datalength] = CLIP(Rc, 0, 255);
    pOutput[i + j * width + 1 * datalength] = CLIP(Gc, 0, 255);
    pOutput[i + j * width + 2 * datalength] = CLIP(Bc, 0, 255);

    return;
}

 void VibranceRGB(
    __global uchar *pInput,
    __global uchar *pOutput,
    float  increment)
{
    int i   = get_global_id(0);
    int j   = get_global_id(1);
    int dim = get_global_id(2);
    int width    = get_global_size(0);
    int height   = get_global_size(1);
    int datalength = width * height;
    float bvalue   = (float)pInput[i + j * width + 2 * datalength];
    float gvalue   = (float)pInput[i + j * width + 1 * datalength];
    float rvalue   = (float)pInput[i + j * width + 0 * datalength];
    float min_val  = MIN_VALUE(rvalue,gvalue,bvalue);
    float max_val  = MAX_VALUE(rvalue,gvalue,bvalue);
    float saturation = max_val - min_val;
    float k = 1.0f + increment * (1.0f - saturation / 255.0f);
    float Y = 0.299f * rvalue + 0.587f * gvalue + 0.114f * bvalue;
    float Rc = Y * (1 - k) + rvalue * k;
    float Gc = Y * (1 - k) + gvalue * k;
    float Bc = Y * (1 - k) + bvalue * k;

    pOutput[i + j * width + 0 * datalength] = (uchar)CLIP(Rc, 0, 255);
    pOutput[i + j * width + 1 * datalength] = (uchar)CLIP(Gc, 0, 255);
    pOutput[i + j * width + 2 * datalength] = (uchar)CLIP(Bc, 0, 255);

    return;
}

__kernel void Vibrance(
    __global void *pInput,
    __global void *pOutput,
    float  increment,
    int format)
{
    if(format == FORMAT_RGB){
        VibranceRGB(pInput, pOutput, increment);
    } else {
        VibranceF32RGB(pInput, pOutput, increment);
    }

    return;
}

