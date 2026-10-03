// CL Kernel Source Code for SurfaceBlur Algorithm
// Version 1.0.0

__kernel void SurfaceBlur(
    __global float* pInput,
    __global float* pOutput,
    __global char* pInd)
{
    #define ABS(x) ((x) > (0) ? (x) : -1 * (x))

    int j = get_global_id(0); //width
    int i = get_global_id(1); //height
    int width = get_global_size(0);
    int height = get_global_size(1);
    int datalength = width * height;
    int thre = 10;
    int radius = 7;

    pOutput[i * width + j] = pInput[i * width + j];
    if (pInd[i * width + j]) {
        if ((i >= radius && i < (height - 1 - radius))
            && (j >= radius && j < (width - 1 - radius))) {
            float p0 = pInput[i * width + j];
            float t1_sum = 0;
            float mask3_sum = 0;
            for (int indexY = i - radius; indexY < (i + radius +1); indexY++) {
                 for (int indexX = j - radius; indexX < (j + radius +1); indexX++) {
                    float aa = pInput[indexY * width + indexX];
                    float mask_2 = 1 - ABS(aa - p0) / (2.5f * thre);
                    float mask_3 = mask_2 * (mask_2 > 0 ? 1:0);
                    float t1 = aa * mask_3;
                    t1_sum += t1;
                    mask3_sum += mask_3;
                }
            }
            pOutput[i * width + j] = t1_sum / mask3_sum;
        } else {
            pOutput[i * width + j] = pInput[i * width + j];
        }

    }
}

