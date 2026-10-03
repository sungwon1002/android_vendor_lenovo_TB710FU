// CL Kernel Source Code for DefoggingMinChannel Algorithm
// Version 1.0.0

enum CL_DATA_TYPE { BI_BYTE, BI_INT, BI_FLOAT };

__kernel void DefoggingMinChannel(
    __global const float* image,
    __global float* minchannelImage,
    int datatype)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int datalength = width * height;
    __global const float* fprtin = image;
    __global float* fptrout = minchannelImage;
    const float R = fprtin[x + y * width];
    const float G = fprtin[x + y * width + datalength];
    const float B = fprtin[x + y * width + 2 * datalength];
    float min = R;
    min = (min < G) ? min : G;
    min = (min < B) ? min : B;
    fptrout[x + y * width] = min;
}