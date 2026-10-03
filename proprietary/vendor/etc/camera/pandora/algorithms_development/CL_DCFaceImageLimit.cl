#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
__kernel void DCFaceImageLimit(__global float *input, __global uchar *output) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int datalength = width * height;
    output[x + y * width] = (uchar)MIN(MAX(input[x + y * width], 0), 255);
    output[x + y * width + datalength] =
        (uchar)MIN(MAX(input[x + y * width + datalength], 0), 255);
    output[x + y * width + datalength * 2] =
        (uchar)MIN(MAX(input[x + y * width + datalength * 2], 0), 255);
}