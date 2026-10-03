// CL Kernel Source Code for DCCubic Algorithm
// Version 1.0.0

#define CALIBRATE_MESH_WIDTH 41

__kernel void DCCubicPreProcessFunc(
    __global float *kx,
    __global float *ky,
    __global int *grid_x,
    __global int *grid_y,
    __global float *source,
    __global float *output)
{
    int j   = get_global_id(0);
    int i   = get_global_id(1);
    int width  = get_global_size(0);
    int height = get_global_size(1);
    int datalength = width * height;

    float temp1[2], temp2[2], temp[2];
    float x, y;

    temp1[0] = (1 - kx[i]) * source[(grid_x[i] * CALIBRATE_MESH_WIDTH + grid_y[j]) * 2] +
        kx[i] * source[((grid_x[i] + 1) * CALIBRATE_MESH_WIDTH + grid_y[j]) * 2];
    temp1[1] = (1 - kx[i]) * source[(grid_x[i] * CALIBRATE_MESH_WIDTH + grid_y[j]) * 2 + 1] +
        kx[i] * source[((grid_x[i] + 1) * CALIBRATE_MESH_WIDTH + grid_y[j]) * 2 + 1];

    temp2[0] = (1 - kx[i]) * source[(grid_x[i] * CALIBRATE_MESH_WIDTH + grid_y[j] + 1) * 2] +
        kx[i] * source[((grid_x[i] + 1) * CALIBRATE_MESH_WIDTH + grid_y[j] + 1) * 2];
    temp2[1] = (1 - kx[i]) * source[(grid_x[i] * CALIBRATE_MESH_WIDTH + grid_y[j] + 1) * 2 + 1] +
        kx[i] * source[((grid_x[i] + 1) * CALIBRATE_MESH_WIDTH + grid_y[j] + 1) * 2 + 1];

    temp[0] = (1 - ky[j]) * temp1[0] + ky[j] * temp2[0];
    temp[1] = (1 - ky[j]) * temp1[1] + ky[j] * temp2[1];

    x = temp[0];
    y = temp[1];
    output[j + i * width] = x;
    output[j + i * width + datalength] = y;
}