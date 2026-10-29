// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cmath>
#include <cstdint>
// Operation order is part of the portable device numerical contract.
// Compile CPU validation with -ffp-contract=off and without -ffast-math.
inline float ternip_sigmoid(float x) {
    float e=std::exp(-std::fabs(x));return x>=0 ? 1.f/(1.f+e) : e/(1.f+e);
}
inline float ternip_swish(float x) {return x*ternip_sigmoid(x);}
inline float ternip_rms_inverse(const float* x,unsigned n,float epsilon) {
    float sum=0.f;
    for(unsigned j=0;j<n;++j) sum+=x[j]*x[j];
    return 1.f/std::sqrt(sum/n+epsilon);
}
inline int8_t ternip_quantized(float value,float multiplier) {
    float z=value*multiplier;
    float rounded=std::copysign(std::floor(std::fabs(z)+.5f),z);
    return int8_t(std::fmax(-128.f,std::fmin(127.f,rounded)));
}
inline float ternip_rescale(int32_t dot,float activation_scale,float weight_scale) {
    return float(dot)*activation_scale*weight_scale;
}
