// CL Kernel Source Code for CropAndResizeBillinearF32RGB Algorithm
// Version 1.0.0

__kernel void CropAndResizeBillinearRGB(
    __global uchar* input_image,
    __global uchar* output_image,
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
    int x1 = crop_x + (int)(x * x_ratio);
    int y1 = crop_y + (int)(y * y_ratio);
    int x2 = (int)(x1 + x_ratio);
    int y2 = (int)(y1 + y_ratio);
    float x_diff = (x * x_ratio) - x1;
    float y_diff = (y * y_ratio) - y1;
    float pixel1 = input_image[y1 * input_width + x1 + dim * datalength];
    float pixel2 = input_image[y1 * input_width + x2 + dim * datalength];
    float pixel3 = input_image[y2 * input_width + x1 + dim * datalength];
    float pixel4 = input_image[y2 * input_width + x2 + dim * datalength];
    float top_pixel = pixel1 * (1 - x_diff) + pixel2 * x_diff;
    float bottom_pixel = pixel3 * (1 - x_diff) + pixel4 * x_diff;

    output_image[y * output_width + x + dim * newDataLength] =
        top_pixel * (1 - y_diff) + bottom_pixel * y_diff;
}
