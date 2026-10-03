// CL Kernel Source Code for SketchGauss Algorithm
// Version 1.0.0

constant int gaussianKernel[7][7] = {
    {0, 3,  6,  8,  6,  3,  0},
    {3, 11, 23, 30, 23, 11, 3},
    {6, 23, 51, 65, 51, 23, 6},
    {8, 30, 65, 84, 65, 30, 8},
    {6, 23, 51, 65, 51, 23, 6},
    {3, 11, 23, 30, 23, 11, 3},
    {0, 3,  6,  8,  6,  3,  0}
};

__kernel void SketchGauss(
    __global float *pInput,
    __global float *pOutput)
{
    int j = get_global_id(0); //width
    int i = get_global_id(1); //height
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int radium = 3;

    // start gaussian blur
    float sum = 0;
    for (int m = -radium; m <= radium; m++) {
        for (int n = -radium; n <= radium; n++) {
            int pi = i + m;
            int pj = j + n;
            if (pi < 0) {
                pi = abs(pi);
            } else if(pi > height - 1) {
                pi = height + height - pi - 2;
            }

            if (pj < 0) {
                pj = abs(pj);
            } else if(pj > width - 1) {
                pj = width + width - pj - 2;
            }

            sum += pInput[pi * width + pj] *
                gaussianKernel[n + radium][m + radium];
       }
    }
    pOutput[i * width + j] = sum /1000;
}