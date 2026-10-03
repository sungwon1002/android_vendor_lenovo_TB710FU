__kernel void DCFaceCentreCrop(__global float *input, __global float *output,
                         int old_width, int old_height, int delta_x,
                         int delta_y)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int new_width = get_global_size(0);
    int new_height = get_global_size(1);

    int new_datalength = new_width * new_height;
    int old_datalength = old_width * old_height;
    int new_index = x + y * new_width;
    int old_index = (x + delta_x) + (y + delta_y) * old_width;

    output[new_index] = input[old_index];
    output[new_index + new_datalength] = input[old_index + old_datalength];
    output[new_index + 2 * new_datalength] = input[old_index + 2 * old_datalength];
}