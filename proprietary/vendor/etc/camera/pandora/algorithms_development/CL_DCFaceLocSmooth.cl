__kernel void DCFaceLocSmooth(__global float* loc, __global float* objn)
{
    int j = get_global_id(0);
    int i = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);
    if (objn[i * w + j] > 1.0f) {
        loc[(i * w + j) * 2] = loc[(i * w + j) * 2] / objn[i * w + j];
        loc[(i * w + j) * 2 + 1] = loc[(i * w + j) * 2 + 1] / objn[i * w + j];
    }
}
