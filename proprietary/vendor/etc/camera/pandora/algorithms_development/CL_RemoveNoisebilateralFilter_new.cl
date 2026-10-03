// CL Kernel Source Code for RemoveNoisebilateralFilter Algorithm
// Version 1.0.0

void getGausssianMask(float *Mask, int w, int h, float spaceSigma)
{
    int center_h = (h - 1) / 2;
    int center_w = (w - 1) / 2;
    float x, y;

    for (int i = 0; i < h; ++i) {
        y = (i - center_h ) * (i - center_h );
        float *Maskdata = Mask + i * w;
        for (int j = 0; j < w; ++j) {
            x = (j - center_w) * (j - center_w);
            float g = exp(-(x + y) / (2 * spaceSigma * spaceSigma));
            Maskdata[j] = g;
        }
    }
}

__kernel void RemoveNoisebilateralFilter_new(
    __global float *Newsrc,
    __global float *dst,
    __global float *colorMask,
    int d,
    float spaceSigma,
    float colorSigma,
    int id)
{
    int J = get_global_id(0);
    int I = get_global_id(1);

    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int base = width * height * id;

    int kwidth =  d;
    int kheight = d;

    int hh = (kheight - 1) / 2;
    int ww = (kwidth - 1) / 2;
    int newwidth = width + d;
    int newheight = height + d;
    int maskSize = kwidth * kheight;
    float spaceMask[100];
    float Mask[100] ;

    getGausssianMask(spaceMask, kwidth, kheight, spaceSigma);

    int i = I + hh;
    int j = J + ww;
    int graydiff= 0.0f;
    float ss = 0.0f;
    float space_color_sum = 0.0f;
    for (int r = -hh; r <= hh; ++r) {
        for (int c = -ww; c <= ww; ++c) {
            int centerPix = Newsrc[i * (width + kwidth) + j];
            int pix = Newsrc[(i + r ) * (width + kwidth) + j + c];
            graydiff = abs(pix - centerPix);
            float colorWeight = colorMask[graydiff % 256];
            Mask[(r + hh) * kwidth + c + ww] = colorWeight * spaceMask[(r + hh) * kwidth + c + ww];
            space_color_sum = space_color_sum + Mask[(r + hh) * kwidth + c + ww];
        }
    }

    for(int m = 0; m < maskSize; ++m) {
        *(Mask + m) /=  space_color_sum;
    }

    for (int r = -hh; r <= hh; ++r) {
        for (int c = -ww; c <= ww; ++c) {
            int pix = Newsrc[(i + r) * (width + kwidth) + j + c];
            float mask = Mask[(r + hh) * kwidth + c + ww];
            ss = ss + mask * pix;
        }
    }
    if (ss < 0) {
        ss = 0;
    }
    else if (ss > 255) {
        ss = 255;
    }
    dst[(I) * width + J] = ss;

}
