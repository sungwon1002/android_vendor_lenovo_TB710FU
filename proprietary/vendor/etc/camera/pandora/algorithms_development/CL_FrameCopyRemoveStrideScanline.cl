// CL Kernel Source Code for FrameCopyRemoveStrideScanline Algorithm
// Version 1.0.0

#define FORMAT_NV21   0 // YVU
#define FORMAT_NV12   1 // YUV
#define FORMAT_YUV_3P 2
#define FORMAT_RGB    9
#define FORMAT_HLS    10
#define FORMAT_F32RGB 11
#define FORMAT_F32HLS 12

__kernel void FrameCopyRemoveStrideScanline(
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
        int gSrcOffset = 1 * stride * scanline;
        int bSrcOffset = 2 * stride * scanline;
        int gDstOffset = 1 * width * height;
        int bDstOffset = 2 * width * height;
        pOutputRGB[j * width + i] = pInputRGB[j * stride + i];
        pOutputRGB[gDstOffset + j * width + i] = pInputRGB[gSrcOffset + j * stride + i];
        pOutputRGB[bDstOffset + j * width + i] = pInputRGB[bSrcOffset + j * stride + i];
    }

    if (format == FORMAT_F32RGB || format == FORMAT_F32HLS) {
        float *pInputRGB  = (float *)pInput;
        float *pOutputRGB = (float *)pOutput;
        int gSrcOffset = 1 * stride * scanline;
        int bSrcOffset = 2 * stride * scanline;
        int gDstOffset = 1 * width * height;
        int bDstOffset = 2 * width * height;
        pOutputRGB[j * width + i] = pInputRGB[j * stride + i];
        pOutputRGB[gDstOffset + j * width + i] = pInputRGB[gSrcOffset + j * stride + i];
        pOutputRGB[bDstOffset + j * width + i] = pInputRGB[bSrcOffset + j * stride + i];
    }

    if (format == FORMAT_YUV_3P) {
        uchar *pInputYUV = (uchar *)pInput;
        uchar *pOutputYUV = (uchar *)pOutput;
        pOutputYUV[j * width + i] = pInputYUV[j * stride + i];
        if (i % 2 == 0 && j % 2 == 0) {
            int uvWidth    = width / 2;
            int uvStride   = stride / 2;
            int uSrcOffset = stride * scanline;
            int vSrcOffset = uSrcOffset + uSrcOffset / 4;
            int uDstOffset = width * height;
            int vDstOffset = uDstOffset + uDstOffset / 4;
            pOutputYUV[uDstOffset + j / 2 * uvWidth + i / 2] = pInputYUV[uSrcOffset + j / 2 * uvStride + i / 2];
            pOutputYUV[vDstOffset + j / 2 * uvWidth + i / 2] = pInputYUV[vSrcOffset + j / 2 * uvStride + i / 2];
        }
    }

    if (format == FORMAT_NV21 || format == FORMAT_NV12) {
        uchar *pInputYUV = (uchar *)pInput;
        uchar *pOutputYUV = (uchar *)pOutput;
        pOutputYUV[j * width + i] = pInputYUV[j * stride + i];
        if (j % 2 == 0) {
            int uvSrcOffset = stride * scanline;
            int uvDstOffset = width * height;
            pOutputYUV[uvDstOffset + j / 2 * width + i] = pInputYUV[uvSrcOffset + j / 2 * stride + i];
            pOutputYUV[uvDstOffset + j / 2 * width + i] = pInputYUV[uvSrcOffset + j / 2 * stride + i];
        }
    }

    return;
}
