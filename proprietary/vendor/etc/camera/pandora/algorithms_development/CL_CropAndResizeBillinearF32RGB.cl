// CL Kernel Source Code for CropAndResizeBillinearF32RGB Algorithm
// Version 1.0.0

__kernel void CropAndResizeBillinearF32RGB(
    __global float* input_image,
    __global float* output_image,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    int crop_x,
    int crop_y,
    int crop_w,
    int crop_h)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int dim = get_global_id(2);

    const int datalength = input_width * input_height;
    const int newDataLength = output_width * output_height;
    float x_ratio = (float)crop_w / (float)output_width;
    float y_ratio = (float)crop_h / (float)output_height;
    float src_x_f = x * x_ratio;
    float src_y_f = y * y_ratio;
    int x1 = crop_x + (int)src_x_f;
    int y1 = crop_y + (int)src_y_f;
    int x2 = min(x1 + 1, input_width - 1);
    int y2 = min(y1 + 1, input_height - 1);
    float x_diff = (crop_x + src_x_f) - x1;
    float y_diff = (crop_y + src_y_f) - y1;
    float pixel1 = input_image[y1 * input_width + x1 + dim * datalength];
    float pixel2 = input_image[y1 * input_width + x2 + dim * datalength];
    float pixel3 = input_image[y2 * input_width + x1 + dim * datalength];
    float pixel4 = input_image[y2 * input_width + x2 + dim * datalength];
    float top_pixel = pixel1 * (1 - x_diff) + pixel2 * x_diff;
    float bottom_pixel = pixel3 * (1 - x_diff) + pixel4 * x_diff;

    output_image[y * output_width + x + dim * newDataLength] =
        top_pixel * (1 - y_diff) + bottom_pixel * y_diff;
}
