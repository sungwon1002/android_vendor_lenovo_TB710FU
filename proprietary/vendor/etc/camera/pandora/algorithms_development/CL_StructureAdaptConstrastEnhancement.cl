// CL Kernel Source Code for StructureAdaptConstrastEnhancement Algorithm
// Version 1.0.0


__kernel void StructureAdaptConstrastEnhancement(
    __global float *yuvBuf,
    __global float *mean,
    __global float *stdDev,
    float maxCfg,
    int stride,
    __global float *globalMean)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width = get_global_size(0);
    int planeOffset = j * width + i;
    int planeOffsetWithStride = j * stride + i;

    float y, u, v;
    float sum = 0;
    float cg = 0;

    y = yuvBuf[planeOffsetWithStride];
    cg = 0.4f * globalMean[0] / stdDev[planeOffset];
    if (cg > maxCfg) {
        cg = maxCfg;
    } else if (cg < 1) {
        cg = 1;
    }
    y = (mean[planeOffset] + cg * (y - mean[planeOffset]) > 255 ? 255 : mean[planeOffset] + cg * (y - mean[planeOffset]));
    y = y < 0 ? 0 : y;
    yuvBuf[planeOffsetWithStride] = y;
}