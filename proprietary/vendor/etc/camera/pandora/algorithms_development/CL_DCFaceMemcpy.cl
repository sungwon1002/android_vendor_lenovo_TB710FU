__kernel void DCFaceMemcpy(__global uchar *input, __global uchar *output)
{
    int index = get_global_id(0);
    output[index] = input[index];
}