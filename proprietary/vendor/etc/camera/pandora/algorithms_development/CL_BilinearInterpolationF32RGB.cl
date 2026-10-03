// CL Kernel Source Code for BilinearInterpolationF32RGB Algorithm
// Version 1.0.0

__kernel void BilinearInterpolationF32RGB(
    __global float* input_image,
    __global float* output_image,
    int input_width,
    int input_height,
    int output_width,
    int output_height)
{
    int j   = get_global_id(0);
    int i   = get_global_id(1);
    int dim = get_global_id(2);
    int inSize    = input_width * input_height;
    int outSize   = output_width * output_height;
    float x_ratio = (float)input_width / (float)output_width;
    float y_ratio = (float)input_height / (float)output_height;
    float inputX  = (j + 0.5f) * x_ratio - 0.5f;
    float inputY  = (i + 0.5f) * y_ratio - 0.5f;
    int x1 = inputX;
    int y1 = inputY;
    int x2 = min(x1 + 1, input_width - 1);
    int y2 = min(y1 + 1, input_height - 1);
    float x_diff = inputX - x1;
    float y_diff = inputY - y1;
    float pixel1 = input_image[y1 * input_width + x1 + dim * inSize];
    float pixel2 = input_image[y1 * input_width + x2 + dim * inSize];
    float pixel3 = input_image[y2 * input_width + x1 + dim * inSize];
    float pixel4 = input_image[y2 * input_width + x2 + dim * inSize];
    float top_pixel    = pixel1 * (1 - x_diff) + pixel2 * x_diff;
    float bottom_pixel = pixel3 * (1 - x_diff) + pixel4 * x_diff;

    output_image[i * output_width + j + dim * outSize] =
        top_pixel * (1 - y_diff) + bottom_pixel * y_diff;
}