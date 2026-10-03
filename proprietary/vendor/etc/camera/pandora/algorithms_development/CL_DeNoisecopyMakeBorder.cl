// CL Kernel Source Code for DeNoisecopyMakeBorder Algorithm
// Version 1.0.0

__kernel void DeNoisecopyMakeBorder(
    __global float* src,
    __global float* dst,
    int ww,
    int hh,
    int id)
{
    const int width = get_global_size(0) - ww;
    const int height = get_global_size(1) -hh;
    int j = get_global_id(0);
    int i = get_global_id(1);
    src = src + width * height * id;
    dst = dst + (width + ww) * (height + hh) * id;

    float sumx = 0;
    float sumy = 0;
    if( i < hh / 2 && j < ww / 2) { //left && top
        int dst_pos = j + i * (width + ww);
        *(dst + dst_pos) = *(src);
    } else if (i < hh / 2 && j >= ww / 2 + width) { //right && top
        int dst_pos = j + i * (width + ww);
        *(dst + dst_pos) = *(src + width - 1);
    } else if (i < hh / 2 && j >= ww / 2 && j <= ww / 2 + width) { //middle && top
        int dst_pos = j + i * (width + ww);
        *(dst + dst_pos) = *(src + j - ww / 2);
    } else if(hh / 2 <= i && i < height + hh / 2  && ww / 2 <= j && j < width + ww / 2) { //center
        dst[i * (width + ww) + j] = src[(i - hh / 2) * (width) + j - ww / 2];
    } else if( i >= hh / 2 + height && j < ww / 2) { //left && bottom
        int dst_pos = j + i * (width + ww);
        *(dst + dst_pos) = *(src + ( height -1) * width);
    } else if (i >= hh / 2 + height && j >= ww / 2 + width) { //right && bottom
        int dst_pos = j + i * (width + ww);
        *(dst + dst_pos) = *(src +  height * width - 1);
    } else if (i >= hh / 2 + height && j >= ww / 2 && j <= ww / 2 + width) { //middle && bottom
        int dst_pos = j + i * (width + ww);
        *(dst + dst_pos) = *(src + j - ww / 2 + (height - 1) * width);
    } else if ( i >= hh / 2 && i <= hh / 2 + height && j < ww / 2) { //middle  && left
        int dst_pos = j + i * (width + ww);
        *(dst + dst_pos) = *(src + (i - hh / 2) * width);
    } else if ( i >= hh / 2 && i <= hh / 2 + height && j >= width + ww / 2) { //middle  && right
        int dst_pos = j + i * (width + ww);
        *(dst + dst_pos) = *(src + (i - hh / 2 + 1) * width - 1);
    }
};
