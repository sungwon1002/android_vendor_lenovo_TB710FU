// CL Kernel Source Code for GaussBlur Algorithm
// Version 1.0.0

__kernel void GaussBlur(
    __global char* pInput,
    __global float* pOutput,
    __global int* gaussianKernel)
{
    int j = get_global_id(0); //width
    int i = get_global_id(1); //height
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int datalength = width * height;
    int radium   = 3;

    // start gaussian blur
    float sum  = 0;
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
           } else if (pj > width - 1) {
               pj = width + width - pj - 2;
           }

           const int weight = gaussianKernel[(n + radium) * 7 + (m + radium)];
           sum += (float)pInput[pi * width + pj] * weight;
       }
    }
    pOutput[i * width + j] = sum / 1000;
}
