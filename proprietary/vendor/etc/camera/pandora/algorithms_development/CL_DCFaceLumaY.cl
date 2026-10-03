__kernel void DCFaceLumaY(__global float* input, __global float* output, __global float* loc)
{
    int j = get_global_id(0);
    int i = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);

    float x, y;
    int x1, y1;
    float k_x, k_y;
    x = loc[(i * w + j) * 2];
    y = loc[(i * w + j) * 2 + 1];
    x = fmax(0, fmin(x, h - 4.0f));
    y = fmax(0, fmin(y, w - 4.0f));
    x1 = trunc(x);
    y1 = trunc(y);
    k_x = x - x1;
    k_y = y - y1;
    output[i * w + j] = (1 - k_x) * (1 - k_y) * input[x1 * w + y1] + (1 - k_x) * k_y * input[x1 * w + y1 + 1] +
        (k_x) * (1 - k_y) * input[(x1 + 1) * w + y1] + k_x * k_y * input[(x1 + 1) * w + y1 + 1];
}
