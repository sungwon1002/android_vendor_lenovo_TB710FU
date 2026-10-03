__kernel void DCFacePreGeometry(__global float *kx,__global float *ky,__global int *grid_x,
    __global int *grid_y, __global float *source, int deta_x, int deta_y, int k,
    __global float *output)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int width = get_global_size(0);
    int height = get_global_size(1);

    int datalength = width * height;

    float temp1[2];
    float temp2[2];
    float temp[2];

    int i = -deta_x * k + y;
    int j = -deta_y * k + x;

    temp1[0] = (1 - kx[i + deta_x * k]) * source[(grid_x[i + deta_x * k] * 47 + grid_y[j + deta_y * k]) * 2]
        + kx[i + deta_x * k] * source[((grid_x[i + deta_x * k] + 1) * 47 + grid_y[j + deta_y * k]) * 2];
    temp1[1] = (1 - kx[i + deta_x * k]) * source[(grid_x[i + deta_x * k] * 47 + grid_y[j + deta_y * k]) * 2 + 1]
        + kx[i + deta_x * k] * source[((grid_x[i + deta_x * k] + 1) * 47 + grid_y[j + deta_y * k]) * 2 + 1];
    temp2[0] = (1 - kx[i + deta_x * k]) * source[(grid_x[i + deta_x * k] * 47 + grid_y[j + deta_y * k] + 1) * 2]
        + kx[i + deta_x * k] * source[((grid_x[i + deta_x * k] + 1) * 47 + grid_y[j + deta_y * k] + 1) * 2];
    temp2[1] = (1 - kx[i + deta_x * k]) * source[(grid_x[i + deta_x * k] * 47 + grid_y[j + deta_y * k] + 1) * 2 + 1]
        + kx[i + deta_x * k] * source[((grid_x[i + deta_x * k] + 1) * 47 + grid_y[j + deta_y * k] + 1) * 2 + 1];

    temp[0] = (1 - ky[j + deta_y * k]) * temp1[0] + ky[j + deta_y * k] * temp2[0];
    temp[1] = (1 - ky[j + deta_y * k]) * temp1[1] + ky[j + deta_y * k] * temp2[1];

    output[x + y * width] = temp[0];
    output[x + y * width + datalength] = temp[1];
}