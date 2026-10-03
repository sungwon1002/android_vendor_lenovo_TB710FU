// CL Kernel Source Code for PaintingProcess Algorithm
// Version 1.0.0

__kernel void PaintingProcess(
    __global float *pInputRGB,
    __global unsigned char *pInputGRAY,
    __global float *pOutput,
    __private const int padding,
    __private const int datalength)
{
    int i = get_global_id(0); // width
    int j = get_global_id(1); // height
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int outputIdxR = j * width + i;
    int outputIdxG = outputIdxR + datalength;
    int outputIdxB = outputIdxG + datalength;

    int res[8] = {0};
    for (int m = -padding; m < padding + 1; m++) {
        for (int n = -padding; n < padding + 1; n++) {
            int pi = i + n;
            int pj = j + m;
            pi = (pi < 0) ? 0 : (pi > width - 1) ? width - 1 : pi;
            pj = (pj < 0) ? 0 : (pj > height - 1) ? height - 1 : pj;
            int val = pInputGRAY[pj * width + pi];
            res[val] += 1;
        }
    }

    int maxNum = res[0];
    int picVal = 0;
    for (int k = 0; k < 8; k++) {
        if (maxNum < res[k]) {
            maxNum = res[k];
            picVal = k;
        }
    }

    bool find  = false;
    for (int m = -padding; m < padding + 1; m++) {
        for (int n = -padding; n < padding + 1; n++) {
            int pi = i + n;
            int pj = j + m;

            if (pi < 0) {
                pi = 0;
            } else if(pi > width - 1) {
                pi = width - 1;
            }

            if (pj < 0) {
                pj = 0;
            } else if(pj > height - 1) {
                pj = height - 1;
            }

            int inputIdxR = pj * width + pi;
            if (pInputGRAY[inputIdxR] == picVal) {
                int inputIdxG = inputIdxR + datalength;
                int inputIdxB = inputIdxG + datalength;
                pOutput[outputIdxR] = pInputRGB[inputIdxR];
                pOutput[outputIdxG] = pInputRGB[inputIdxG];
                pOutput[outputIdxB] = pInputRGB[inputIdxB];
                break;
            }
        }
    }
}
