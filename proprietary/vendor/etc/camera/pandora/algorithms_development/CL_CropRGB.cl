// CL Kernel Source Code for Crop Algorithm
// Version 1.0.0

__kernel void CropRGB(
    __global const unsigned char * input,
    int input_width,
    int input_height,
    __global unsigned char * output,
    int output_width,
    int output_height,
    int crop_x,
    int crop_y)
{
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);
    int dim = get_global_id(2);

    if (gid_x < output_width && gid_y < output_height) {
        int input_x = crop_x + gid_x;
        int input_y = crop_y + gid_y;
        int in_size = input_width * input_height;
        int out_size = output_width * output_height;

        if (input_x >= 0 && input_x < input_width &&
            input_y >= 0 && input_y < input_height) {
            int input_index = input_y * input_width + input_x + dim * in_size;
            int output_index = gid_y * output_width + gid_x + dim * out_size;
            output[output_index] = input[input_index];
        }
    }
}
