__kernel void DCFaceWarpPerspectiveY(__global float* outputLoc, __global float* outputn,
    int w, int h, int loc_in_i, int loc_in_j, int loc_out_i, int loc_out_j, int M_i, int M_j,
    __global float *M) {
    int j = get_global_id(0);
    int i = get_global_id(1);
    float result  = 0.0f;
    M = M + (M_i * 46 + M_j) * 3 * 3;
    float A = M[2 * 3] * j - M[0];
    float B = M[2 * 3 + 1] * j - M[1];
    float C = M[2] - j;
    float D = M[2 * 3 + 0] * i - M[1 * 3 + 0];
    float K = M[2 * 3 + 1] * i - M[1 * 3 + 1];
    float H = M[1 * 3 + 2] - i;
    float x = (C * D - A * H) / (B * D - A * K) + 1;
    float y = (C - B * x) / A + 1;
    int i_loc = i + loc_out_i;
    int j_loc = j + loc_out_j;
    outputLoc[(i_loc * w + j_loc) * 2] = x + loc_in_i + outputLoc[(i_loc * w + j_loc) * 2];
    outputLoc[(i_loc * w + j_loc) * 2 + 1] = y + loc_in_j + outputLoc[(i_loc * w + j_loc) * 2 + 1];
    outputn[i_loc * w + j_loc] = outputn[i_loc * w + j_loc] + 1;
}