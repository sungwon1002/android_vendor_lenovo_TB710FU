__kernel void DCFaceCentreCropUV(__global float *input, __global uchar *output,
                         int old_width, int old_height, int delta_x,
                         int delta_y,int stride, int scanline)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int new_width = get_global_size(0);
    int new_height = get_global_size(1);

    int new_datalength = stride * scanline;
    int old_datalength = old_width * old_height;
    int new_index = (x + y * stride) * 2;
    int old_index = ((x + delta_x) + (y + delta_y) * old_width) * 2;

    output[new_index + new_datalength * 4] = trunc(fmax(fmin(input[old_index + old_datalength * 4],255.0f),0.0f));
    output[new_index + 1 + new_datalength * 4] = trunc(fmax(fmin(input[old_index + 1 + old_datalength * 4],255.0f),0.0f));
}