// CL Kernel Source Code for Rgb2Hls Algorithm
// Version 1.0.0

#define FORMAT_RGB    9

void DistortionStep3RGB(
    __global float *pIn,
    __global float *n_val,
    __global uchar *pOut)
{
    int j = get_global_id(0);
    int i = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int frameSize = width * height;
    int temp_h = height + 3;
    int temp_w = width + 3;
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

    if (n[i * temp_w + j] <= 0){
        for(int index = 0; index < 3; index++) {
            __global float *img1Plain = img1 + temp_w * temp_h * index;
            float luma_z[4] = {
                img1Plain[i * temp_w + (j - 1)],
                img1Plain[i * temp_w + (j + 1)],
                img1Plain[(i - 1) * temp_w + j],
                img1Plain[(i + 1) * temp_w + j]
            };
            float n_z[4] = {
                n[i * temp_w + (j - 1)],
                n[i * temp_w + (j + 1)],
                n[(i - 1) * temp_w + j],
                n[(i + 1) * temp_w + j]
            };
            int count = 4;
            for (int z = 0; z < 4; z++) {
                if (n_z[z] == 0){
                    count --;
                } else{
                    img1Plain[i * temp_w + j] = img1Plain[i * temp_w + j] +luma_z[z] / n_z[z];
                }
            }
            img1Plain[i * temp_w + j] = img1Plain[i * temp_w + j] / count;

        }
        outputR[(i - 1) * width + (j - 1)] = (uchar)img1R[i * temp_w + j];
        outputG[(i - 1) * width + (j - 1)] = (uchar)img1G[i * temp_w + j];
        outputB[(i - 1) * width + (j - 1)] = (uchar)img1B[i * temp_w + j];
    }

    return;
}

void DistortionStep3F32RGB(
    __global float *pIn,
    __global float *n_val,
    __global float *pOut)
{
    int j = get_global_id(0);
    int i = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int frameSize = width * height;
    int temp_h = height + 3;
    int temp_w = width + 3;
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

    if (n[i * temp_w + j] <= 0){
        for(int index = 0; index < 3; index++) {
            __global float *img1Plain = img1 + temp_w * temp_h * index;
            float luma_z[4] = {
                img1Plain[i * temp_w + (j - 1)],
                img1Plain[i * temp_w + (j + 1)],
                img1Plain[(i - 1) * temp_w + j],
                img1Plain[(i + 1) * temp_w + j]
            };
            float n_z[4] = {
                n[i * temp_w + (j - 1)],
                n[i * temp_w + (j + 1)],
                n[(i - 1) * temp_w + j],
                n[(i + 1) * temp_w + j]
            };
            int count = 4;
            for (int z = 0; z < 4; z++) {
                if (n_z[z] == 0){
                    count --;
                } else{
                    img1Plain[i * temp_w + j] = img1Plain[i * temp_w + j] +luma_z[z] / n_z[z];
                }
            }
            img1Plain[i * temp_w + j] = img1Plain[i * temp_w + j] / count;

        }
        outputR[(i - 1) * width + (j - 1)] = img1R[i * temp_w + j];
        outputG[(i - 1) * width + (j - 1)] = img1G[i * temp_w + j];
        outputB[(i - 1) * width + (j - 1)] = img1B[i * temp_w + j];
    }

    return;
}
__kernel void DistortionStep3(
    __global float *pIn,
    __global float *n_val,
    __global void *pOut,
    int format)
{
    if(format == FORMAT_RGB){
        DistortionStep3RGB(pIn, n_val, pOut);
    } else {
        DistortionStep3F32RGB(pIn, n_val, pOut);
    }

    return;
}
