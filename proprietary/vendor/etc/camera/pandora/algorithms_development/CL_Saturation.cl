// CL Kernel Source Code for Saturation Algorithm
// Version 1.0.0
#define FORMAT_RGB    9

#define MAX(a, b)   ((a) > (b) ? (a) : (b))
#define MIN(a, b)   ((a) < (b) ? (a) : (b))
#define CLIP(x,a,b) (MIN(MAX(x,a),b))

void SaturationF32RGB(
    __global float* pInput,
    __global float* pOutput,
    float S)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int dim = get_global_id(2);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int datalength = width * height;
    float bvalue = pInput[i + j * width + 2 * datalength];
    float gvalue = pInput[i + j * width + 1 * datalength];
    float rvalue = pInput[i + j * width + 0 * datalength];
    float Y = 0.299f * rvalue + 0.587f * gvalue + 0.114f * bvalue;
    float Rc = Y * (1 - S) + rvalue * S;
    float Gc = Y * (1 - S) + gvalue * S;
    float Bc = Y * (1 - S) + bvalue * S;
    pOutput[i + j * width + 0 * datalength] = CLIP(Rc, 0, 255);
    pOutput[i + j * width + 1 * datalength] = CLIP(Gc, 0, 255);
    pOutput[i + j * width + 2 * datalength] = CLIP(Bc, 0, 255);

    return;
}

 void SaturationRGB(
    __global uchar *pInput,
    __global uchar *pOutput,
    float S)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int dim = get_global_id(2);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int datalength = width * height;
    float bvalue = (float)pInput[i + j * width + 2 * datalength];
    float gvalue = (float)pInput[i + j * width + 1 * datalength];
    float rvalue = (float)pInput[i + j * width + 0 * datalength];
    float Y = 0.299f * rvalue + 0.587f * gvalue + 0.114f * bvalue;
    float Rc = Y * (1 - S) + rvalue * S;
    float Gc = Y * (1 - S) + gvalue * S;
    float Bc = Y * (1 - S) + bvalue * S;
    pOutput[i + j * width + 0 * datalength] = (uchar)CLIP(Rc, 0, 255);
    pOutput[i + j * width + 1 * datalength] = (uchar)CLIP(Gc, 0, 255);
    pOutput[i + j * width + 2 * datalength] = (uchar)CLIP(Bc, 0, 255);

    return;
}

__kernel void Saturation(
    __global void *pInput,
    __global void *pOutput,
    float S,
    int format)
{
    if(format == FORMAT_RGB){
        SaturationRGB(pInput, pOutput, S);
    } else {
        SaturationF32RGB(pInput, pOutput, S);
    }

    return;
}

