__kernel void DCFaceGeometryCali(__global float *img, __global float *input,
    __global float *output, int image_width, int image_height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int datalength = width * height;
    int image_datalength = image_width * image_height;

    float w1[11] = {0.0f, -0.081f,-0.128f,-0.147f,-0.144f,-0.125f,-0.096f,-0.063f, 0.032f,-0.009f, 0.0f};
    float w2[11] = {1.0f,  0.981f, 0.928f, 0.847f, 0.744f, 0.625f, 0.496f, 0.363f, 0.232f, 0.109f, 0.0f};
    float w3[11] = {0.0f, 0.1089f,0.2319f, 0.363f, 0.496f, 0.625f, 0.744f, 0.847f, 0.928f, 0.981f, 1.0f};
    float w4[11] = {0.0f,-0.0089f,-0.0319f,-0.063f,-0.096f,-0.125f,-0.144f,-0.147f,-0.128f,-0.081f,0.0f};

    float xf = input[x + y * width];
    float yf = input[x + y * width + datalength];

    if ((xf >= 1.0f) && (yf >= 1.0f) && (xf <= image_height - 3) && (yf <= image_width - 3)){
        int x1= (int)(xf);
        int y1= (int)(yf);
        int x2 = (int)((xf - x1) * 10);
        int y2 = (int)((yf - y1) * 10);
        float W = w1[x2] * w2[y2] + w1[x2] * w3[y2] + w2[x2] * w1[y2] + w2[x2] * w2[y2]
            + w2[x2] * w3[y2] + w2[x2] * w4[y2] + w3[x2] * w1[y2] + w3[x2] * w2[y2]
            + w3[x2] * w3[y2] + w3[x2] * w4[y2] + w4[x2] * w2[y2] + w4[x2] * w3[y2];
        float luma1[3];

        luma1[0] = img[(x1 - 1) * image_width + y1] * w1[x2] * w2[y2]
            + img[(x1 - 1) * image_width + y1 + 1] * w1[x2] * w3[y2]
            + img[x1 * image_width + y1 - 1] * w2[x2] * w1[y2]
            + img[x1 * image_width + y1] * w2[x2] * w2[y2]
            + img[x1 * image_width + y1 + 1] * w2[x2] * w3[y2]
            + img[x1 * image_width + y1 + 2] * w2[x2] * w4[y2]
            + img[(x1 + 1) * image_width + y1 - 1] * w3[x2] * w1[y2]
            + img[(x1 + 1) * image_width + y1] * w3[x2] * w2[y2]
            + img[(x1 + 1) * image_width + y1 + 1] * w3[x2] * w3[y2]
            + img[(x1 + 1) * image_width + y1 + 2] * w3[x2] * w4[y2]
            + img[(x1 + 2) * image_width + y1] * w4[x2] * w2[y2]
            + img[(x1 + 2) * image_width + y1 + 1] * w4[x2] * w3[y2];

        luma1[1] = img[(x1 - 1) * image_width + y1 + image_datalength] * w1[x2] * w2[y2]
            + img[(x1 - 1) * image_width + y1 + 1 + image_datalength] * w1[x2] * w3[y2]
            + img[x1 * image_width + y1 - 1 + image_datalength] * w2[x2] * w1[y2]
            + img[x1 * image_width + y1 + image_datalength] * w2[x2] * w2[y2]
            + img[x1 * image_width + y1 + 1 + image_datalength] * w2[x2] * w3[y2]
            + img[x1 * image_width + y1 + 2 + image_datalength] * w2[x2] * w4[y2]
            + img[(x1 + 1) * image_width + y1 - 1 + image_datalength] * w3[x2] * w1[y2]
            + img[(x1 + 1) * image_width + y1 + image_datalength] * w3[x2] * w2[y2]
            + img[(x1 + 1) * image_width + y1 + 1 + image_datalength] * w3[x2] * w3[y2]
            + img[(x1 + 1) * image_width + y1 + 2 + image_datalength] * w3[x2] * w4[y2]
            + img[(x1 + 2) * image_width + y1 + image_datalength] * w4[x2] * w2[y2]
            + img[(x1 + 2) * image_width + y1 + 1 + image_datalength] * w4[x2] * w3[y2];

        luma1[2] = img[(x1 - 1) * image_width + y1 + image_datalength * 2] * w1[x2] * w2[y2]
            + img[(x1 - 1) * image_width + y1 + 1 + image_datalength * 2] * w1[x2] * w3[y2]
            + img[x1 * image_width + y1 - 1 + image_datalength * 2] * w2[x2] * w1[y2]
            + img[x1 * image_width + y1 + image_datalength * 2] * w2[x2] * w2[y2]
            + img[x1 * image_width + y1 + 1 + image_datalength * 2] * w2[x2] * w3[y2]
            + img[x1 * image_width + y1 + 2 + image_datalength * 2] * w2[x2] * w4[y2]
            + img[(x1 + 1) * image_width + y1 - 1 + image_datalength * 2] * w3[x2] * w1[y2]
            + img[(x1 + 1) * image_width + y1 + image_datalength * 2] * w3[x2] * w2[y2]
            + img[(x1 + 1) * image_width + y1 + 1 + image_datalength * 2] * w3[x2] * w3[y2]
            + img[(x1 + 1) * image_width + y1 + 2 + image_datalength * 2] * w3[x2] * w4[y2]
            + img[(x1 + 2) * image_width + y1 + image_datalength * 2] * w4[x2] * w2[y2]
            + img[(x1 + 2) * image_width + y1 + 1 + image_datalength * 2] * w4[x2] * w3[y2];

        output[y * width + x] = luma1[0] / W;
        output[y * width + x + datalength] = luma1[1] / W;
        output[y * width + x + 2 * datalength] = luma1[2] / W;
    }

}