__kernel void DCFaceRectangleY(__global float* img, int width, int height,
int x1, int y1, int x2, int y2, uchar Y, int thickness) {
    int x = get_global_id(0) + x1;
    int y = get_global_id(1) + y1;

    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;
    int datalength = width * height;
    int channels = 3;
    if (thickness > 1 && ((x >= x1 && x <= x1 + thickness - 1)
        || (x >= x2 - thickness + 1 && x <= x2)
        || (y >= y1 && y <= y1 + thickness - 1)
        || (y >= y2 - thickness + 1 && y <= y2))) {
        img[idx] = Y;
    }
}