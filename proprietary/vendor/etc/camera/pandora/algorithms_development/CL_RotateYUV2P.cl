// CL Kernel Source Code for Rotate Algorithm
// Version 1.0.0
void rotatePlain(
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

void rotateUVPlain(
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
    if( j < width && i < height) {
        int xc = width / 2;
        int yc = height / 2;

        int dx = j-xc;
        int dy = i-yc;
        int xpos =  dx * cosTheta + dy * sinTheta + xc;
        int ypos = -dx * sinTheta + dy * cosTheta + yc;

        if(xpos >= 0 && ypos >= 0 && xpos < width && ypos < height) {
            int inputIndex = ypos * stride + xpos;
            int outputIndex = i * stride + j;
            outputIndex = outputIndex * 2;
            inputIndex = inputIndex * 2;
            output[outputIndex] = input[inputIndex];
            output[outputIndex + 1] = input[inputIndex + 1];
        } else {
            int outputIndex = i * stride + j;
            outputIndex = outputIndex * 2;
            output[outputIndex] = 128;
            output[outputIndex + 1] = 128;
        }
    }
}

__kernel void RotateYUV2P(
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
    rotatePlain(input, output, width, height,
        stride, scanline, i, j, cosTheta, sinTheta);
    rotateUVPlain(input + dataSize, output + dataSize, width / 2, height / 2,
        stride / 2, scanline / 2,i, j, cosTheta, sinTheta);
}
