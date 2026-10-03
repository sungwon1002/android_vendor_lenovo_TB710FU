// CL Kernel Source Code for Exposure Algorithm
// Version 1.0.0
// S = 255 / (255 - light)

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CLIP(x,a,b) (MIN(MAX(x,a),b))

__kernel void ExposureRGB(
    __global uchar* pInput,
    __global uchar* pOutput,
    float S)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int width      = get_global_size(0);
    int height     = get_global_size(1);
    int datalength = width * height;

    float bvalue = pInput[i + j * width + 2 * datalength];
    float gvalue = pInput[i + j * width + 1 * datalength];
    float rvalue = pInput[i + j * width + 0 * datalength];

    float Rc = rvalue * S;
    float Gc = gvalue * S;
    float Bc = bvalue * S;
    pOutput[i + j * width + 0 * datalength] = CLIP(Rc, 0, 255);
    pOutput[i + j * width + 1 * datalength] = CLIP(Gc, 0, 255);
    pOutput[i + j * width + 2 * datalength] = CLIP(Bc, 0, 255);
}
