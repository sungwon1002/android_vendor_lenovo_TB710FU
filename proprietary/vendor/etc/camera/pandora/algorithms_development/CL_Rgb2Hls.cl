// CL Kernel Source Code for Rgb2Hls Algorithm
// Version 1.0.0

__kernel void Rgb2Hls(
    __global float *pIn,
    __global float *pOut)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int frameSize = width * height;
    int planeOffset = j * width + i;

    float r = pIn[planeOffset];
    float g = pIn[frameSize + planeOffset];
    float b = pIn[frameSize * 2 + planeOffset];
    float dr = (float)r / 255;
    float dg = (float)g / 255;
    float db = (float)b / 255;
    float tmp = dg < db ? db : dg;
    float cmax = tmp < dr ? dr : tmp;
    tmp = dg < db ? dg : db;
    float cmin = tmp < dr ? tmp : dr;
    float cdes = cmax - cmin;
    float hh, ll, ss;

    ll = (cmax + cmin) / 2;
    if (cdes) {
        if (ll <= 0.5f) {
            ss = (cmax - cmin) / (cmax + cmin);
        } else {
            ss = (cmax - cmin) / (2 - cmax - cmin);
        }
        if (cmax == dr) {
            hh = (0 + (dg - db) / cdes) * 60;
        } else if (cmax == dg) {
            hh = (2 + (db - dr) / cdes) * 60;
        } else {
            hh = (4 + (dr - dg) / cdes) * 60;
        }
        if (hh < 0) {
            hh += 360;
        }
    } else {
       hh = ss = 0;
    }
    if (hh > 360) hh = 360;
    if (ll > 1) ll = 1;
    if (ss > 1) ss = 1;
    if (hh < 0) hh = 0;
    if (ll < 0) ll = 0;
    if (ss < 0) ss = 0;

    pOut[planeOffset] = hh;
    pOut[frameSize + planeOffset] = ll;
    pOut[frameSize * 2 + planeOffset] = ss;
}