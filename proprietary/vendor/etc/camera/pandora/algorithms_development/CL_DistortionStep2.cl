// CL Kernel Source Code for Rgb2Hls Algorithm
// Version 1.0.0

#define FORMAT_RGB    9

void DistortionStep2F32RGB(
    __global float *pIn,
    __global float *n_val,
    __global float *pOut)
{
    int j = get_global_id(0);
    int i = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int frameSize = width * height;
    int temp_h   = height + 3;
    int temp_w   = width + 3;
    __global float *img1  = pIn;
    __global float *img1R = img1;
    __global float *img1G = img1 + temp_w * temp_h;
    __global float *img1B = img1 + temp_w * temp_h * 2;
    __global float *n     = n_val;
    __global float *outImg  = pOut;
    __global float *outputR = outImg;
    __global float *outputG = outImg + width * height;
    __global float *outputB = outImg + width * height * 2;

    i = i + 1;
    j = j + 1;
    if (n[i * temp_w + j] > 0) {
        img1R[i * temp_w + j] = img1R[i * temp_w + j] / n[i * temp_w + j];
        img1G[i * temp_w + j] = img1G[i * temp_w + j] / n[i * temp_w + j];
        img1B[i * temp_w + j] = img1B[i * temp_w + j] / n[i * temp_w + j];
        n[i * temp_w + j] = 1;

        outputR[(i - 1) * width + (j - 1)] = img1R[i * temp_w + j];
        outputG[(i - 1) * width + (j - 1)] = img1G[i * temp_w + j];
        outputB[(i - 1) * width + (j - 1)] = img1B[i * temp_w + j];
    }
}
void DistortionStep2RGB(
    __global float *pIn,
    __global float *n_val,
    __global uchar *pOut)
{
    int j = get_global_id(0);
    int i = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int frameSize = width * height;
    int temp_h   = height + 3;
    int temp_w   = width + 3;
    __global float *img1  = pIn;
    __global float *img1R = img1;
    __global float *img1G = img1 + temp_w * temp_h;
    __global float *img1B = img1 + temp_w * temp_h * 2;
    __global float *n     = n_val;
    __global uchar *outImg  = pOut;
    __global uchar *outputR = outImg;
    __global uchar *outputG = outImg + width * height;
    __global uchar *outputB = outImg + width * height * 2;

    i = i + 1;
    j = j + 1;
    if (n[i * temp_w + j] > 0) {
        img1R[i * temp_w + j] = img1R[i * temp_w + j] / n[i * temp_w + j];
        img1G[i * temp_w + j] = img1G[i * temp_w + j] / n[i * temp_w + j];
        img1B[i * temp_w + j] = img1B[i * temp_w + j] / n[i * temp_w + j];
        n[i * temp_w + j] = 1;

        outputR[(i - 1) * width + (j - 1)] = (uchar)img1R[i * temp_w + j];
        outputG[(i - 1) * width + (j - 1)] = (uchar)img1G[i * temp_w + j];
        outputB[(i - 1) * width + (j - 1)] = (uchar)img1B[i * temp_w + j];
    }
}

__kernel void DistortionStep2(
    __global float *pIn,
    __global float *n_val,
    __global void *pOut,
    int format)
{
    if(format == FORMAT_RGB){
        DistortionStep2RGB(pIn, n_val, pOut);
    } else {
        DistortionStep2F32RGB(pIn, n_val, pOut);
    }

    return;
}
