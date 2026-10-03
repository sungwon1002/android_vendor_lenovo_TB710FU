// CL Kernel Source Code for Rotate Algorithm
// Version 1.0.0

__kernel void Rotate(
    __global float *input,
    __global float *output,
    int width,
    int height,
    float cosTheta,
    float sinTheta)
{
    int j   = get_global_id(0);
    int i   = get_global_id(1);
    int dim = get_global_id(2);
    int dataSize = width * height;

    int xc = width / 2;
    int yc = height / 2;

    int dx = j-xc;
    int dy = i-yc;
    int xpos =  dx * cosTheta + dy * sinTheta + xc;
    int ypos = -dx * sinTheta + dy * cosTheta + yc;

    if(xpos >= 0 && ypos >= 0 && xpos < width && ypos < height) {
        output[i * width + j + dim * dataSize] =
            input[ypos * width + xpos + dim * dataSize];
    } else {
        output[i * width + j + dim * dataSize] = 0;
    }
}
