// CL Kernel Source Code for VignetteCorner Algorithm
// Version 1.0.0


__kernel void VignetteCornerF32RGB(
    __global float *pInput,
    __global float *pOutput,
    __global float *mRatioK,
    int mStrength,
    float mStartX,
    float mEndX,
    float mStartY,
    float mEndY,
    float mK1,
    float mK2,
    int datalength
    )
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int dim    = get_global_id(2);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int jwidth = j * width;
    float bvalue = pInput[i + jwidth + (datalength << 1)];
    float gvalue = pInput[i + jwidth + datalength];
    float rvalue = pInput[i + jwidth ];
    float patch  = 1.0f;
    int maxlen   = width > height ? width : height;
    int cx = (maxlen / 2);
    int cy = cx;
    float maxDist = sqrt((float)(maxlen * cx));
    int  jmaxlen = maxlen * j;
    int nx = jmaxlen / width;
    int ny = jmaxlen / height;
    float curDist = ((nx - cx) * (nx - cx) + (ny - cy) * (ny - cy));
    curDist = sqrt(curDist);
    curDist = curDist / maxDist;
    float t;

    if (curDist < mStartX ) {
        t = 1 + mK1 * curDist;
    } else if (mStartX < curDist && curDist < mEndX) {
        t = mRatioK[0] * curDist * curDist * curDist +
            mRatioK[1] * curDist * curDist +
            mRatioK[2] * curDist +
            mRatioK[3];
    } else {
        t = mEndY + mK2 * (curDist - mEndX);
    }
    if (mStrength < 50) {
        patch = 255;
    }
    float dpatch = (1 - t) * patch;
    float out = rvalue * t + dpatch;
    int base = i + j * width;
    out = out > 255 ? 255 : out;
    out = out < 0 ? 0 : out;
    pOutput[base] = out;
    out = gvalue * t + dpatch;
    out = out > 255 ? 255 : out;
    out = out < 0 ? 0 : out;
    pOutput[base + datalength] = out;
    out = bvalue * t + dpatch;
    out = out > 255 ? 255 : out;
    out = out < 0 ? 0 : out;
    pOutput[base + (datalength << 1)] = out;
}
