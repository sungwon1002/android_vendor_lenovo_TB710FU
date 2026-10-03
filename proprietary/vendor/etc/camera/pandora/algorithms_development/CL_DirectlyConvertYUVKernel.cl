// CL Kernel Source Code for FrameCopyRemoveStrideScanline Algorithm
// Version 1.0.0

#define FORMAT_NV21   0 // YVU
#define FORMAT_NV12   1 // YUV
#define FORMAT_YUV_3P 2

__kernel void DirectlyConvertYUVKernel(
    __global uchar *pFrame,
    __global uchar *uv,
    int origFormat,
    int newFormat,
    int wxh,
    int wxhD4)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int widthD2  = get_global_size(0);
    int heightD2 = get_global_size(1);

    if (origFormat == FORMAT_NV21 && newFormat == FORMAT_YUV_3P) {
        __global uchar *pInputUV = uv;
        __global uchar *pOutputU = pFrame + wxh;
        __global uchar *pOutputV = pOutputU + wxhD4;
        int uoffset  = y * widthD2 + x;
        int uvoffset = y * widthD2 * 2 + x * 2;
        pOutputV[uoffset] = pInputUV[uvoffset];
        pOutputU[uoffset] = pInputUV[uvoffset + 1];
    }

    if (origFormat == FORMAT_NV12 && newFormat == FORMAT_YUV_3P) {
        __global uchar *pInputUV = uv;
        __global uchar *pOutputU = pFrame + wxh;
        __global uchar *pOutputV = pOutputU + wxhD4;
        int uoffset  = y * widthD2 + x;
        int uvoffset = y * widthD2 * 2 + x * 2;
        pOutputU[uoffset] = pInputUV[uvoffset];
        pOutputV[uoffset] = pInputUV[uvoffset + 1];
    }

    if (origFormat == FORMAT_YUV_3P && newFormat == FORMAT_NV21) {
        __global uchar *pInputU = uv;
        __global uchar *pInputV = uv + wxhD4;
        __global uchar *pOutputUV = pFrame + wxh;
        int uvoffset = y * widthD2 * 2 + x * 2;
        int uoffset  = y * widthD2 + x;
        pOutputUV[uvoffset] = pInputV[uoffset];
        pOutputUV[uvoffset + 1] = pInputU[uoffset];
    }

    if (origFormat == FORMAT_YUV_3P && newFormat == FORMAT_NV12) {
        __global uchar *pInputU = uv;
        __global uchar *pInputV = uv + wxhD4;
        __global uchar *pOutputUV = pFrame + wxh;
        int uvoffset = y * widthD2 * 2 + x * 2;
        int uoffset  = y * widthD2 + x;
        pOutputUV[uvoffset] = pInputU[uoffset];
        pOutputUV[uvoffset + 1] = pInputV[uoffset];
    }

    if (origFormat == FORMAT_NV21 && newFormat == FORMAT_NV12) {
        __global uchar *pInputUV  = uv;
        __global uchar *pOutputUV = pFrame + wxh;
        int offset = y * widthD2 * 2 + x * 2;
        pOutputUV[offset] = pInputUV[offset + 1];
        pOutputUV[offset + 1] = pInputUV[offset];
    }

    if (origFormat == FORMAT_NV12 && newFormat == FORMAT_NV21) {
        __global uchar *pInputUV  = uv;
        __global uchar *pOutputUV = pFrame + wxh;
        int offset = y * widthD2 * 2 + x * 2;
        pOutputUV[offset] = pInputUV[offset + 1];
        pOutputUV[offset + 1] = pInputUV[offset];
    }

    return;
}
