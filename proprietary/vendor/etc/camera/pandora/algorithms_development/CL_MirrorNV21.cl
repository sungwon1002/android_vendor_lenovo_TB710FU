__kernel void MirrorNV21(
    __global unsigned char* src,
    __global unsigned char* dst,
    int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int src_index = y * width + x;
    int dst_index = y * width + (width - x - 1);

    // Handle Y plane
    dst[dst_index] = src[src_index];

    // Handle UV plane
    if (y % 2 == 0 && x % 2 == 0) {
        int uv_src_index = width * height + y * width / 2 + x;
        int uv_dst_index = width * height + y * width / 2 + (width - x - 2);

        dst[uv_dst_index] = src[uv_src_index];  // U
        dst[uv_dst_index + 1] = src[uv_src_index + 1];  // V
    }
}

