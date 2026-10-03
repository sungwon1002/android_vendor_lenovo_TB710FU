// CL Kernel Source Code for Apply3dLut Algorithm
// Version 1.0.0

__kernel void Apply3dLutUchar(
    __global uchar* input_image,
    __global uchar* output_image,
    __global float* lut,
    int lut_size,
    float k)
{
    const int gid_x = get_global_id(0);
    const int gid_y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int datalength = width * height;
    float colors[8][3];

    const float R = input_image[gid_x + gid_y * width];
    const float G = input_image[gid_x + gid_y * width + datalength];
    const float B = input_image[gid_x + gid_y * width + 2 * datalength];

    float rcache = (float)R / 255.0f * (lut_size - 1);
    float gcache = (float)G / 255.0f * (lut_size - 1);
    float bcache = (float)B / 255.0f * (lut_size - 1);

    int index_r_high = (int)rcache;
    int index_g_high = (int)gcache;
    int index_b_high = (int)bcache;

    int index_r_highinc =
        (index_r_high >= lut_size - 1) ? lut_size - 1 : index_r_high + 1;
    int index_g_highinc =
        (index_g_high >= lut_size - 1) ? lut_size - 1 : index_g_high + 1;
    int index_b_highinc =
        (index_b_high >= lut_size - 1) ? lut_size - 1 : index_b_high + 1;

    float coef_r_low = rcache - index_r_high;
    float coef_g_low = gcache - index_g_high;
    float coef_b_low = bcache - index_b_high;

    float c00[3], c01[3], c10[3], c11[3], c0[3], c1[3];
    float Rout, Gout, Bout;

    // I000
    colors[0][0] = lut[((index_b_high * lut_size + index_g_high) *
        lut_size + index_r_high) * 3];
    colors[0][1] = lut[((index_b_high * lut_size + index_g_high) *
        lut_size + index_r_high) * 3 + 1];
    colors[0][2] = lut[((index_b_high * lut_size + index_g_high) *
        lut_size + index_r_high) * 3 + 2];

    // I100
    colors[1][0] = lut[((index_b_high * lut_size + index_g_high) *
        lut_size + index_r_highinc) * 3];
    colors[1][1] = lut[((index_b_high * lut_size + index_g_high) *
        lut_size + index_r_highinc) * 3 + 1];
    colors[1][2] = lut[((index_b_high * lut_size + index_g_high) *
        lut_size + index_r_highinc) * 3 + 2];

    // I010
    colors[2][0] = lut[((index_b_high * lut_size + index_g_highinc) *
        lut_size + index_r_high) * 3];
    colors[2][1] = lut[((index_b_high * lut_size + index_g_highinc) *
        lut_size + index_r_high) * 3 + 1];
    colors[2][2] = lut[((index_b_high * lut_size + index_g_highinc) *
        lut_size + index_r_high) * 3 + 2];

    // I110
    colors[3][0] = lut[((index_b_high * lut_size + index_g_highinc) *
         lut_size + index_r_highinc) * 3];
    colors[3][1] = lut[((index_b_high * lut_size + index_g_highinc) *
         lut_size + index_r_highinc) * 3 + 1];
    colors[3][2] = lut[((index_b_high * lut_size + index_g_highinc) *
         lut_size + index_r_highinc) * 3 + 2];

    // I001
    colors[4][0] = lut[((index_b_highinc * lut_size + index_g_high) *
         lut_size + index_r_high) * 3];
    colors[4][1] = lut[((index_b_highinc * lut_size + index_g_high) *
         lut_size + index_r_high) * 3 + 1];
    colors[4][2] = lut[((index_b_highinc * lut_size + index_g_high) *
         lut_size + index_r_high) * 3 + 2];

    // I101
    colors[5][0] = lut[((index_b_highinc * lut_size + index_g_high) *
        lut_size + index_r_highinc) * 3];
    colors[5][1] = lut[((index_b_highinc * lut_size + index_g_high) *
        lut_size + index_r_highinc) * 3 + 1];
    colors[5][2] = lut[((index_b_highinc * lut_size + index_g_high) *
        lut_size + index_r_highinc) * 3 + 2];

    // I011
    colors[6][0] = lut[((index_b_highinc * lut_size + index_g_highinc) *
        lut_size + index_r_high) * 3];
    colors[6][1] = lut[((index_b_highinc * lut_size + index_g_highinc) *
        lut_size + index_r_high) * 3 + 1];
    colors[6][2] = lut[((index_b_highinc * lut_size + index_g_highinc) *
        lut_size + index_r_high) * 3 + 2];

    // I111
    colors[7][0] = lut[((index_b_highinc * lut_size + index_g_highinc) *
        lut_size + index_r_highinc) * 3];
    colors[7][1] = lut[((index_b_highinc * lut_size + index_g_highinc) *
        lut_size + index_r_highinc) * 3 + 1];
    colors[7][2] = lut[((index_b_highinc * lut_size + index_g_highinc) *
        lut_size + index_r_highinc) * 3 + 2];

    c00[0] = (1 - coef_r_low) * colors[0][0] + coef_r_low * colors[1][0];
    c00[1] = (1 - coef_r_low) * colors[0][1] + coef_r_low * colors[1][1];
    c00[2] = (1 - coef_r_low) * colors[0][2] + coef_r_low * colors[1][2];

    c01[0] = (1 - coef_r_low) * colors[4][0] + coef_r_low * colors[5][0];
    c01[1] = (1 - coef_r_low) * colors[4][1] + coef_r_low * colors[5][1];
    c01[2] = (1 - coef_r_low) * colors[4][2] + coef_r_low * colors[5][2];

    c10[0] = (1 - coef_r_low) * colors[2][0] + coef_r_low * colors[3][0];
    c10[1] = (1 - coef_r_low) * colors[2][1] + coef_r_low * colors[3][1];
    c10[2] = (1 - coef_r_low) * colors[2][2] + coef_r_low * colors[3][2];

    c11[0] = (1 - coef_r_low) * colors[6][0] + coef_r_low * colors[7][0];
    c11[1] = (1 - coef_r_low) * colors[6][1] + coef_r_low * colors[7][1];
    c11[2] = (1 - coef_r_low) * colors[6][2] + coef_r_low * colors[7][2];

    c0[0] = (1 - coef_g_low) * c00[0] + coef_g_low * c10[0];
    c0[1] = (1 - coef_g_low) * c00[1] + coef_g_low * c10[1];
    c0[2] = (1 - coef_g_low) * c00[2] + coef_g_low * c10[2];

    c1[0] = (1 - coef_g_low) * c01[0] + coef_g_low * c11[0];
    c1[1] = (1 - coef_g_low) * c01[1] + coef_g_low * c11[1];
    c1[2] = (1 - coef_g_low) * c01[2] + coef_g_low * c11[2];

    // R,G,B
    Rout = k * ((1 - coef_b_low) * c0[0] + coef_b_low * c1[0]) * 255 + (1.0f - k) * R;
    Gout = k * ((1 - coef_b_low) * c0[1] + coef_b_low * c1[1]) * 255 + (1.0f - k) * G;
    Bout = k * ((1 - coef_b_low) * c0[2] + coef_b_low * c1[2]) * 255 + (1.0f - k) * B;

    output_image[gid_x + gid_y * width] = trunc(min(max(Rout, 0.0f), 255.0f));
    output_image[gid_x + gid_y * width + datalength] = trunc(min(max(Gout, 0.0f), 255.0f));
    output_image[gid_x + gid_y * width + 2 * datalength] = trunc(min(max(Bout, 0.0f), 255.0f));
}
