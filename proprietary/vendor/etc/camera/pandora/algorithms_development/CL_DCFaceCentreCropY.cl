__kernel void DCFaceCentreCropY(__global float *input, __global uchar *output,
                         int old_width, int old_height, int delta_x,
                         int delta_y, int stride)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int new_index = x + y * stride;
    int old_index = (x + delta_x) + (y + delta_y) * old_width;

    output[new_index] = trunc(fmax(fmin(input[old_index],255.0f),0.0f));
}