// CL Kernel Source Code for HlsChangeStep1 Algorithm
// Version 1.0.0


__kernel void HlsChangeStep1(
    __global float *pIn,
    __global float *pHue1,
    __global char *pInd,
    __global float *pLut,
    int8 colorInfo)
{
    int j = get_global_id(0);
    int i = get_global_id(1);
    int width   = get_global_size(0);
    int height  = get_global_size(1);
    int frameSize = width * height;

    #define MAX(a, b)   ((a) > (b) ? (a) : (b))
    #define MIN(a, b)   ((a) < (b) ? (a) : (b))
    #define CLIP(x,a,b) (MIN(MAX(x,a),b))
    #define ABS(x)      ((x) > (0) ? (x) : -1 * (x))
    #define LUT_WIDTH   (81)

    int center   = colorInfo.s0;
    int range1   = colorInfo.s1;
    int strength = colorInfo.s2;
    int low      = colorInfo.s3;
    int high     = colorInfo.s4;
    int is_low_below_high = colorInfo.s5;
    int distance = 0;
    float ratio  = 0;

    float h_value = pIn[i * width + j];
    ratio = 0;
    pInd[i * width + j] = 0;
    if (is_low_below_high) {
        if (h_value > low && h_value < high) {
            distance = ABS((h_value - center));
            pInd[i * width + j] = 1;
        }
    } else {
        /* Right pixels*/
        if (h_value >= 0 && h_value < high) {
            if (center > 180 && center < 360) {
                distance = ABS(h_value + 360 - center);
            } else {
                distance = ABS(h_value - center);
            }
            pInd[i * width + j] = 1;
        } else if (h_value > low && h_value <= 360) { /* Left pixels*/
            if (center > 180 && center < 360) {
                distance = ABS(h_value - center);
            } else {
                distance = ABS(360 - h_value + center);
            }
            pInd[i * width + j] = 1;
        }
    }

    if (pInd[i * width + j]) {
        int index = distance;
        switch(range1) {
            case 20:
                ratio = pLut[0 * LUT_WIDTH + index];
                break;
            case 30:
                ratio = pLut[1 * LUT_WIDTH + index];
                break;
            case 40:
                ratio = pLut[2 * LUT_WIDTH + index];
                break;
            case 50:
                ratio = pLut[3 * LUT_WIDTH + index];
                break;
            case 60:
                ratio = pLut[4 * LUT_WIDTH + index];
                break;
            default:
                ratio = pLut[5 * LUT_WIDTH + index];
                break;
        }
        pHue1[i * width + j] =  h_value + strength * ratio;
        pInd[i * width + j] = 1;
    } else {
        pHue1[i * width + j] =  h_value;
        pInd[i * width + j] = 0;
    }
}
