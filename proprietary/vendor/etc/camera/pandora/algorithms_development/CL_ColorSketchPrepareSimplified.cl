// CL Kernel Source Code for ColorSketchPrepareSimplified Algorithm
// Version 1.0.0

constant int gColorSketchLut[256] =
{
    20,  20,  21,  22,  22,  23,  23,  24,  25,  25,  26,  27,  27,  28,  28,  29,
    30,  30,  31,  31,  32,  33,  33,  34,  35,  35,  36,  37,  37,  38,  39,  39,
    40,  41,  41,  42,  43,  43,  44,  45,  45,  46,  47,  48,  48,  49,  50,  51,
    51,  52,  53,  54,  54,  55,  56,  57,  58,  59,  59,  60,  61,  62,  63,  64,
    65,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
    80,  81,  82,  83,  84,  84,  85,  86,  87,  88,  90,  91,  92,  93,  94,  95,
    96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 202, 203, 204, 205, 206, 207, 208,
    209, 210, 211, 212, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    223, 224, 225, 226, 226, 227, 228, 229, 229, 230, 230, 231, 232, 232, 233, 233,
    234, 234, 235, 235, 236, 236, 237, 237, 238, 238, 239, 239, 239, 240, 240, 241
};

#define MIN_ITEM(A,B) (A>B? B:A)
#define MIN_VALUE(A,B,C) MIN_ITEM(MIN_ITEM(A,B),C)
#define MAX_ITEM(A,B) (A>B? A:B)
#define MAX_VALUE(A,B,C) MAX_ITEM(MAX_ITEM(A,B),C)

__kernel void ColorSketchPrepareSimplified(
    __global float *pInputRGB,
    __global float *pInvRGB,
    __private const int datalength)
{
    int j = get_global_id(0); //width
    int i = get_global_id(1); //height
    int width = get_global_size(0);
    int height = get_global_size(1);
    int indexR = i * width + j;
    int indexG = indexR + datalength;
    int indexB = indexG + datalength;

    // change saturation start
    float b = pInputRGB[indexB];
    float g = pInputRGB[indexG];
    float r = pInputRGB[indexR];
    float luma = 0.2990f * r + 0.5870f * g + 0.1140f * b;
    float imgMax = MAX_VALUE(r, g, b);
    float imgMin = MIN_VALUE(r, g ,b);
    float saturation = imgMax - imgMin;

    float k = 2.0f - saturation / 255.0f;
    float lumaWight = 1.0f - k;
    float rNew = luma * lumaWight + r * k;
    float gNew = luma * lumaWight + g * k;
    float bNew = luma * lumaWight + b * k;

    rNew = (rNew > 255.0f) ? 255.0f : (rNew < 0) ? 0 : rNew;
    gNew = (gNew > 255.0f) ? 255.0f : (gNew < 0) ? 0 : gNew;
    bNew = (bNew > 255.0f) ? 255.0f : (bNew < 0) ? 0 : bNew;
    // change saturation end

    // change_brightness start
    pInputRGB[indexR] =
        (float)(gColorSketchLut[(int)(rNew + 0.5f)]);
    pInputRGB[indexG] =
        (float)(gColorSketchLut[(int)(gNew + 0.5f)]);
    pInputRGB[indexB] =
        (float)(gColorSketchLut[(int)(bNew + 0.5f)]);
    // change_brightness end

    pInvRGB[indexR] = 255 - pInputRGB[indexR];
    pInvRGB[indexG] = 255 - pInputRGB[indexG];
    pInvRGB[indexB] = 255 - pInputRGB[indexB];
}
