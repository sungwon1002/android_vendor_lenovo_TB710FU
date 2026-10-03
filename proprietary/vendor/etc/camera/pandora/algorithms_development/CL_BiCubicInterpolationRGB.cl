// CL Kernel Source Code for BiCubicInterpolationRGB Algorithm
// Version 1.0.0

float cubicWeight(float x)
{
    if (x < 0.0f) {
        x = -x;
    }
    float absX = x;
    float absX2 = absX * absX;
    float absX3 = absX2 * absX;
    float a = -0.5;

    if (absX <= 1) {
        return (a + 2) * absX3 - (a + 3) * absX2 + 1;
    } else if (absX < 2) {
        return a * absX3 - 5 * a * absX2 + 8 * a * absX - 4 * a;
    }

    return 0;
}

__kernel void BiCubicInterpolationRGB(
    __global unsigned char *input,
    __global unsigned char *output,
    int width,
    int height,
    int newWidth,
    int newHeight)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int dim = get_global_id(2);

    float xRatio = (float)width / newWidth;
    float yRatio = (float)height / newHeight;
    float px = 0.0f, py = 0.0f, dx  = 0.0f, dy = 0.0f, weight = 0.0f;
    int outSize = newWidth * newHeight;
    int inSize  = width * height;
    float sum   = 0.0f;

    int pixel = j * newWidth + i;
    px = i * xRatio;
    py = j * yRatio;

    for (dx = -1; dx <= 2; dx++) {
        for (dy = -1; dy <= 2; dy++) {
            int x = (int) (px + dx);
            int y = (int) (py + dy);
            if (x >= 0 && x < width && y >= 0 && y < height) {
                weight = cubicWeight(dx) * cubicWeight(dy);
                sum += input[y * width + x + dim * inSize] * weight;
            }
        }
    }

    output[pixel + dim * outSize] = sum;
}
