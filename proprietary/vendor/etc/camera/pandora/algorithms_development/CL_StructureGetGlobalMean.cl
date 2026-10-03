// CL Kernel Source Code for StructureGetGlobalMean Algorithm
// Version 1.0.0

__kernel void StructureGetGlobalMean(
    __global const  float *input,
    __global float *globalMean,
    __global float  *partialSum,
    __local float *sum,
    __local float *localItemSum,
    uint ySize,
    uint groupCount)
{
    uint localId = get_local_id(0);
    uint groupSize = get_local_size(0);

    localItemSum[localId] = input[get_global_id(0)];
    for (uint index = groupSize / 2; index > 0; index /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId < index) {
            localItemSum[localId] += localItemSum[localId + index];
        }
    }
    if (localId == 0) {
        partialSum[get_group_id(0)] = localItemSum[0];
    }
    if (get_group_id(0) == groupCount - 1 &&
        localId == 0) {
        for (int i = 0; i < groupCount; i++) {
            *sum += partialSum[i];
        }
        *globalMean = *sum / ySize;
    }
};
