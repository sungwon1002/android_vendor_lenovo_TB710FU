// CL Kernel Source Code for Rotate Algorithm
// Version 1.0.0
void rotatePlainY(
    __global unsigned char *input,
    __global unsigned char *output,
    int width,
    int height,
    int stride,
    int scanline,
    int i,
    int j,
    float cosTheta,
    float sinTheta)
{
    int xc = width / 2;
    int yc = height / 2;

    int dx = j-xc;
    int dy = i-yc;
    int xpos =  dx * cosTheta + dy * sinTheta + xc;
    int ypos = -dx * sinTheta + dy * cosTheta + yc;

    if(xpos >= 0 && ypos >= 0 && xpos < width && ypos < height) {
        output[i * stride + j] = input[ypos * stride + xpos];
    } else {
        output[i * stride + j] = 0;
    }
}

void rotatePlainU(
    __global unsigned char *input,
    __global unsigned char *output,
    int width,
    int height,
    int stride,
    int scanline,
    int i,
    int j,
    float cosTheta,
    float sinTheta)
{
    int xc = width / 2;
    int yc = height / 2;

    int dx = j-xc;
    int dy = i-yc;
    int xpos =  dx * cosTheta + dy * sinTheta + xc;
    int ypos = -dx * sinTheta + dy * cosTheta + yc;

    if(xpos >= 0 && ypos >= 0 && xpos < width && ypos < height) {
        output[i * stride + j] = input[ypos * stride + xpos];
    } else {
        output[i * stride + j] = 128;
    }
}

void rotatePlainV(
    __global unsigned char *input,
    __global unsigned char *output,
    int width,
    int height,
    int stride,
    int scanline,
    int i,
    int j,
    float cosTheta,
    float sinTheta)
{
    int xc = width / 2;
    int yc = height / 2;

    int dx = j-xc;
    int dy = i-yc;
    int xpos =  dx * cosTheta + dy * sinTheta + xc;
    int ypos = -dx * sinTheta + dy * cosTheta + yc;

    if(xpos >= 0 && ypos >= 0 && xpos < width && ypos < height) {
        output[i * stride + j] = input[ypos * stride + xpos];
    } else {
        output[i * stride + j] = 128;
    }
}


__kernel void RotateYUV3P(
    __global unsigned char *input,
    __global unsigned char *output,
    int width,
    int height,
    int stride,
    int scanline,
    float cosTheta,
    float sinTheta)
{
    int j   = get_global_id(0);
    int i   = get_global_id(1);
    int dim = get_global_id(2);
    int dataSize = stride * scanline;
    rotatePlainY(input, output, width, height,
        stride, scanline, i, j, cosTheta, sinTheta);
    rotatePlainU(input + dataSize, output + dataSize, width / 2, height / 2,
        stride / 2, scanline / 2, i / 2, j / 2, cosTheta, sinTheta);
    rotatePlainV(input + dataSize * 5 / 4, output + dataSize * 5 / 4, width / 2, height / 2,
        stride / 2, scanline / 2, i / 2, j / 2, cosTheta, sinTheta);
}
