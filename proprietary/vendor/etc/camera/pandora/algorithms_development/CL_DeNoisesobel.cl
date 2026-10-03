// CL Kernel Source Code for DeNoisesobel Algorithm
// Version 1.0.0

__kernel void DeNoisesobel(
    __global const float* input,
    __global float* output,
    int isX,
    int isY,
    int id)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    int base = width * height * id;

    const int gx[3][3] = {{-1, 0, 1}, {-2 , 0, 2}, {-1, 0, 1}};
    const int gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2 ,- 1}};

    if (i >= width || j >= height) {
        return;
    }

    float sumx = 0;
    float sumy = 0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            int px = x + i;
            int py = y + j;
            if (px < 0) {
              px = abs(px);
            } else if (px > width - 1) {
              px = width + width - px - 2;
            }

            if (py < 0) {
              py = abs(py);
            } else if (py > height - 1) {
              py = height + height - py - 2;
            }
            if(isX) {
                sumx += input[py * width + px] *gx[x + 1][y + 1];
            }
            if(isY) {
                sumy += input[py * width + px] *gy[x + 1][y + 1];
            }
        }
    }
    if( isX && isY) {
        sumx = sqrt(sumx * sumx + sumy * sumy);
        output[j * width + i] = sumx;
    }
    else if(isY) {
        output[j * width + i] = sumy;
    }
    else {
        output[j * width + i] = sumx;
    }
};
