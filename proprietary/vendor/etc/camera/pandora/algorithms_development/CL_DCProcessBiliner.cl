// CL Kernel Source Code for DCKernelFunc Algorithm
// Version 1.0.0
__kernel void DCProcessBiliner(__global const float* inputBuffer, __global float* outputBuffer,
    int inputWidth, int outputWidth, float zoomFactor, int cropW, int cropH)
{
    int2 outputCoords = (int2)(get_global_id(0), get_global_id(1));
    int width = get_global_size(0);
    int height = get_global_size(1);
    float2 inputCoords = (float2)((outputCoords.x - width / 2.0f) / zoomFactor + width / 2.0f,
        (outputCoords.y - height / 2.0f) / zoomFactor + height / 2.0f);

    float2 topLeft = floor(inputCoords);
    float2 topRight = (float2)(topLeft.x + 1.0f, topLeft.y);
    float2 bottomLeft = (float2)(topLeft.x, topLeft.y + 1.0f);
    float2 bottomRight = (float2)(topLeft.x + 1.0f, topLeft.y + 1.0f);

    float2 fractional = inputCoords - topLeft;

    int inputIndex = (int)topLeft.y * inputWidth + (int)topLeft.x;

    float topPixelx = inputBuffer[inputIndex * 2];
    float topRightPixelx = inputBuffer[(inputIndex + 1) * 2];
    float bottomLeftPixelx = inputBuffer[(inputIndex + inputWidth) * 2];
    float bottomRightPixelx = inputBuffer[(inputIndex + inputWidth + 1) * 2];

    float topPixely = inputBuffer[inputIndex * 2 + 1];
    float topRightPixely = inputBuffer[(inputIndex + 1) * 2 + 1];
    float bottomLeftPixely = inputBuffer[(inputIndex + inputWidth) * 2 + 1];
    float bottomRightPixely = inputBuffer[(inputIndex + inputWidth + 1) * 2 + 1];

    float interpolatedPixelx = mix(mix(topPixelx, topRightPixelx, fractional.x), mix(bottomLeftPixelx, bottomRightPixelx, fractional.x), fractional.y);
    float interpolatedPixely = mix(mix(topPixely, topRightPixely, fractional.x), mix(bottomLeftPixely, bottomRightPixely, fractional.x), fractional.y);

    int outputIndex = outputCoords.y * outputWidth + outputCoords.x;
    interpolatedPixelx = (interpolatedPixelx - width / 2) * zoomFactor + width / 2;
    interpolatedPixely = (interpolatedPixely - height / 2) * zoomFactor + height / 2;
    interpolatedPixelx = (interpolatedPixelx + cropW) * width / (width + 2 * cropW);
    interpolatedPixely = (interpolatedPixely + cropH) * height / (height + 2 * cropH);
    interpolatedPixelx = fmin(fmax(0.0f, interpolatedPixelx), width - 2.0f);
    interpolatedPixely = fmin(fmax(0.0f, interpolatedPixely), height - 2.0f);

    outputBuffer[outputIndex * 2] = interpolatedPixelx;
    outputBuffer[outputIndex * 2 + 1] = interpolatedPixely;
}