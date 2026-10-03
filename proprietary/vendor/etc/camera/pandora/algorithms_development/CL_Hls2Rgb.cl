// CL Kernel Source Code for Hls2Rgb Algorithm
// Version 1.0.0

__kernel void Hls2Rgb(
    __global float *pIn,
    __global float *pOut)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width   = get_global_size(0);
    int height  = get_global_size(1);
    int frameSize = width * height;
    float h = pIn[width * j + i];
    float l = pIn[width * j + i + frameSize];
    float s = pIn[width * j + i + frameSize * 2];
    float cmax, cmin;

    if (l <= 0.5f) {
       cmax = l * (1 + s);
    } else {
       cmax = l * (1 - s) + s;
    }
    cmin = 2 * l - cmax;
    if (s > -0.000001f && s < 0.000001f) {
        pOut[width * j + i] = l * 255;
        pOut[frameSize + width * j + i] = l * 255;
        pOut[frameSize * 2 + width * j + i] = l * 255;
    } else {
        float hue = 0;
        float rgbVal = 0;
        for (int k = 0; k < 3; k++) {
            if (k == 0) {
                hue = h + 120;
            } else if (k == 1) {
                hue = h;
            } else if (k == 2) {
                hue =  h - 120;
            }
            if (hue > 360) {
                hue -= 360;
            } else if (hue < 0) {
                hue += 360;
            }
            if (hue < 60) {
                rgbVal = (cmin + (cmax - cmin) * hue / 60);
            } else if (hue < 180) {
                rgbVal = cmax;
            } else if (hue < 240) {
                rgbVal = (cmin + (cmax - cmin) * (240 - hue) / 60);
            } else {
                rgbVal = cmin;
            }
            pOut[frameSize * k + width * j + i] = rgbVal * 255;
        }
    }
}