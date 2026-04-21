#pragma once

#include <simd/simd.h>

struct PoseConstants {
    simd_float4x4 projectionMatrix;
    simd_float4x4 viewMatrix;
};

struct InstanceConstants {
    simd_float4x4 modelMatrix;
};
