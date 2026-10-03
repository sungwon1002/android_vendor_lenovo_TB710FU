// CL Kernel Source Code for StructureGetVarianceMean Algorithm
// Version 1.0.0

__kernel void StructureGetVarianceMean(
    __global float *yuvBuf,
    __global float *mean,
    __global float *stdDev,
    int stride)
{
    int j = get_global_id(0);
    int i = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int winSize  = 25;
    float sum = 0, devSum = 0;
    const float CLEpsinon = 0.00001f;

    for (int m = - winSize / 2; m < winSize / 2 + 1; m++) {
        for (int n = - winSize / 2; n < winSize / 2 + 1; n++) {
            int pi = i + m;
            int pj = j + n;
            pi = (pi < 0) ? -pi : (pi > height - 1) ? height - 1 - (pi - height) : pi;
            pj = (pj < 0) ? -pj : (pj > width - 1) ? width - 1 - (pj - width) : pj;
            sum += *(yuvBuf + pi * stride + pj);
        }
    }
    *(mean + i * width + j) = sum / (winSize * winSize);
    if (*(mean + i * width + j) <= 0) {
        *(mean + i * width + j) = CLEpsinon;
    }
    for (int m = - winSize / 2; m < winSize / 2 + 1; m++) {
        for (int n = - winSize / 2; n < winSize / 2 + 1; n++) {
            int pi = i + m;
            int pj = j + n;
            pi = (pi < 0) ? -pi : (pi > height - 1) ? height - 1 - (pi - height) : pi;
            pj = (pj < 0) ? -pj : (pj > width - 1) ? width - 1 - (pj - width) : pj;
            devSum += (*(yuvBuf + pi * stride + pj) -
                       *(mean + i * width + j)) * (*(yuvBuf + pi * stride + pj) -
                       *(mean + i * width + j));
        }
    }
    *(stdDev + i * width + j) = native_sqrt(devSum / (winSize * winSize));
    if (*(stdDev + i * width + j) <= 0) {
        *(stdDev + i * width + j) = CLEpsinon;
    }
}