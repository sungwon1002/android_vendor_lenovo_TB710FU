__kernel void DCFaceWarpPerspectiveUV(__global float* input, __global float* output,
    int width, int height, int x1, int y1, int x2, int y2, int M_i, int M_j,
    __global float *M, int w, int h) {
    int j = get_global_id(0);
    int i = get_global_id(1);
    float resultV  = 0.0f;
    float resultU  = 0.0f;
    M = M + (M_i * 46 + M_j) * 3 * 3;
    float A = M[2 * 3] * j - M[0];
    float B = M[2 * 3 + 1] * j - M[1];
    float C = M[2] - j;
    float D = M[2 * 3 + 0] * i - M[1 * 3 + 0];
    float K = M[2 * 3 + 1] * i - M[1 * 3 + 1];
    float H = M[1 * 3 + 2] - i;
    float x = (C * D - A * H) / (B * D - A * K) + 1;
    float y = (C - B * x) / A + 1;
    int xI = (int)x;
    int yI = (int)y;
    float kx = fabs(x - xI);
    float ky = fabs(y - yI);
    int datalength = width * height * 4;
    int index1 = (xI - 1 + y1) * width + yI - 1 + x1;
    int index2 = (xI - 1 + y1) * width + yI + x1;
    int index3 = (xI + y1) * width + yI - 1 + x1;
    int index4 = (xI + y1) * width + yI + x1;
    if (xI > 0 && yI > 0 && xI <= h + 1 && yI <= w + 1) {
        resultV = input[index1 * 2 + datalength] * (1 - kx) * (1 - ky) +
            input[index2 * 2 + datalength] * (1 - kx) * ky + input[index3 * 2 + datalength] * kx * (1 - ky) +
            input[index4 * 2 + datalength] * kx * ky;
        resultU = input[index1 * 2 + datalength + 1] * (1 - kx) * (1 - ky) +
            input[index2 * 2 + datalength + 1] * (1 - kx) * ky + input[index3 * 2 + datalength + 1] * kx * (1 - ky) +
            input[index4 * 2 + datalength + 1] * kx * ky;
    } else if (xI == 0 && yI == 0) {
        resultV = (1 - kx) * (1 - ky) + (1 - kx) * ky + kx * (1 - ky)
            + input[index4 * 2 + datalength] * kx * ky;
        resultU = (1 - kx) * (1 - ky) + (1 - kx) * ky + kx * (1 - ky)
            + input[index4 * 2 + datalength + 1] * kx * ky;
    } else if (xI == 0) {
        resultV = (1 - kx) * (1 - ky) + (1 - kx) * ky + input[index3 * 2 + datalength] * kx * (1 - ky) +
            input[index4 * 2 + datalength] * kx * ky;
        resultU = (1 - kx) * (1 - ky) + (1 - kx) * ky + input[index3 * 2 + datalength + 1] * kx * (1 - ky) +
            input[index4 * 2 + datalength + 1] * kx * ky;
    } else if (yI == 0) {
        resultV = (1 - kx) * (1 - ky) + input[index2 * 2 + datalength] * (1 - kx) * ky + kx * (1 - ky) +
            input[index4 * 2 + datalength] * kx * ky;
        resultU = (1 - kx) * (1 - ky) + input[index2 * 2 + datalength + 1] * (1 - kx) * ky + kx * (1 - ky) +
            input[index4 * 2 + datalength + 1] * kx * ky;
    }
    output[((i + y2) * width + j + x2) * 2 + datalength] = fmax(resultV, output[((i + y2) * width + j + x2) * 2 + datalength]);
    output[((i + y2) * width + j + x2) * 2 + 1 + datalength] = fmax(resultU, output[((i + y2) * width + j + x2) * 2 + 1 + datalength]);
}