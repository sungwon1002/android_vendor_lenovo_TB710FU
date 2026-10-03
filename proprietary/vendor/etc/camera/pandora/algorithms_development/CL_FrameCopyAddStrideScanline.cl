// CL Kernel Source Code for FrameCopyAddStrideScanline Algorithm
// Version 1.0.0

#define FORMAT_NV21   0 // YVU
#define FORMAT_NV12   1 // YUV
#define FORMAT_YUV_3P 2
#define FORMAT_RGB    9
#define FORMAT_HLS    10
#define FORMAT_F32RGB 11
#define FORMAT_F32HLS 12

__kernel void FrameCopyAddStrideScanline(
    __global void *pInput,
    int format,
    __global void *pOutput,
    int stride,
    int scanline)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);

    if (format == FORMAT_RGB || format == FORMAT_HLS) {
        uchar *pInputRGB  = (float *)pInput;
        uchar *pOutputRGB = (float *)pOutput;
        int gSrcOffset = 1 * width * height;
        int bSrcOffset = 2 * width * height;
        int gDstOffset = 1 * stride * scanline;
        int bDstOffset = 2 * stride * scanline;
        pOutputRGB[j * stride + i] = pInputRGB[j * width + i];
        pOutputRGB[gDstOffset + j * stride + i] = pInputRGB[gSrcOffset + j * width + i];
        pOutputRGB[bDstOffset + j * stride + i] = pInputRGB[bSrcOffset + j * width + i];
    }

    if (format == FORMAT_F32RGB || format == FORMAT_F32HLS) {
        float *pInputRGB  = (float *)pInput;
        float *pOutputRGB = (float *)pOutput;
        int gSrcOffset = 1 * width * height;
        int bSrcOffset = 2 * width * height;
        int gDstOffset = 1 * stride * scanline;
        int bDstOffset = 2 * stride * scanline;
        pOutputRGB[j * stride + i] = pInputRGB[j * width + i];
        pOutputRGB[gDstOffset + j * stride + i] = pInputRGB[gSrcOffset + j * width + i];
        pOutputRGB[bDstOffset + j * stride + i] = pInputRGB[bSrcOffset + j * width + i];
    }

    if (format == FORMAT_YUV_3P) {
        uchar *pInputYUV = (uchar *)pInput;
        uchar *pOutputYUV = (uchar *)pOutput;
        pOutputYUV[j * stride + i] = pInputYUV[j * width + i];
        if (i % 2 == 0 && j % 2 == 0) {
            int uvWidth    = width / 2;
            int uvStride   = stride / 2;
            int uSrcOffset = width * height;
            int vSrcOffset = uSrcOffset + uSrcOffset / 4;
            int uDstOffset = stride * scanline;
            int vDstOffset = uDstOffset + uDstOffset / 4;
            pOutputYUV[uDstOffset + j / 2 * uvStride + i / 2] = pInputYUV[uSrcOffset + j / 2 * uvWidth + i / 2];
            pOutputYUV[vDstOffset + j / 2 * uvStride + i / 2] = pInputYUV[vSrcOffset + j / 2 * uvWidth + i / 2];
        }
    }

    if (format == FORMAT_NV21 || format == FORMAT_NV12) {
        uchar *pInputYUV = (uchar *)pInput;
        uchar *pOutputYUV = (uchar *)pOutput;
        pOutputYUV[j * stride + i] = pInputYUV[j * width + i];
        if (j % 2 == 0) {
            int uvSrcOffset = width * height;
            int uvDstOffset = stride * scanline;
            pOutputYUV[uvDstOffset + j / 2 * stride + i] = pInputYUV[uvSrcOffset + j / 2 * width + i];
            pOutputYUV[uvDstOffset + j / 2 * stride + i] = pInputYUV[uvSrcOffset + j / 2 * width + i];
        }
    }

    return;
}
