// SPDX-License-Identifier: BSD-3-Clause
// Shared-controller FP32 arithmetic for accelerated numerical qualification.
// CUDA is used only for exact integer dot products by the Python caller.
#include "scalar_math.hpp"
extern "C" void ternip_norm_many(const float* x,const float* w,float* y,unsigned rows,unsigned n,float eps) {
#pragma omp parallel for
    for(unsigned r=0;r<rows;++r) {
        float inv=ternip_rms_inverse(x+r*n,n,eps);
        for(unsigned j=0;j<n;++j) y[r*n+j]=x[r*n+j]*inv*w[j];
    }
}
extern "C" void ternip_quant_many(const float* x,const float* norm,int8_t* q,float* scales,unsigned rows,unsigned n) {
#pragma omp parallel for
    for(unsigned r=0;r<rows;++r) {
        float normalized[8192];
        float inv=ternip_rms_inverse(x+r*n,n,1e-6f),maxabs=1e-5f;
        for(unsigned j=0;j<n;++j) {
            float value=x[r*n+j]*inv*norm[j];normalized[j]=value;
            maxabs=std::fmax(maxabs,std::fabs(value));
        }
        float multiplier=127.f/maxabs;scales[r]=1.f/multiplier;
        for(unsigned j=0;j<n;++j) q[r*n+j]=ternip_quantized(normalized[j],multiplier);
    }
}
extern "C" void ternip_rescale_many(const int32_t* dot,const float* scales,float* y,unsigned rows,unsigned n,float ws) {
#pragma omp parallel for
    for(unsigned r=0;r<rows;++r) for(unsigned j=0;j<n;++j) y[r*n+j]=ternip_rescale(dot[r*n+j],scales[r],ws);
}
extern "C" void ternip_recurrent_many(const float* i,const float* f,float* y,float* state,unsigned rows,unsigned n) {
#pragma omp parallel for
    for(unsigned j=0;j<n;++j) {
        float h=0.f;
        for(unsigned r=0;r<rows;++r) {
            unsigned k=r*n+j;h=f[k]*h+(1.f-f[k])*i[k];y[k]=h;
        }
        state[j]=h;
    }
}
extern "C" void ternip_vector_many(unsigned op,const float* x,const float* b,const float* p,float* y,unsigned rows,unsigned n) {
#pragma omp parallel for
    for(unsigned r=0;r<rows;++r) for(unsigned j=0;j<n;++j) {
        unsigned k=r*n+j;
        if(op==4) y[k]=ternip_sigmoid(x[k]);
        else if(op==5) y[k]=ternip_swish(x[k]);
        else if(op==6) y[k]=p[j]+(1.f-p[j])*x[k];
        else if(op==8) y[k]=ternip_swish(x[r*2*n+j])*x[r*2*n+n+j];
        else if(op==9) y[k]=x[k]+b[k];
        else if(op==10) y[k]=x[k]*b[k];
    }
}
