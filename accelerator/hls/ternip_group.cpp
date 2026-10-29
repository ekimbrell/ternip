// SPDX-License-Identifier: BSD-3-Clause
// Resident model/vector controller. TERNIP_RTL_TILES offloads integer tiles to
// the real ternip_core lanes through the versioned tile stream protocol.
// The default backend retains a portable software dot product for comparison.
// Neither backend is throughput-qualified until implementation and card tests.
#include "ternip_group.hpp"
#include "scalar_math.hpp"
#include <cmath>
#include <cstring>
#include <limits>
static float as_float(uint32_t bits) { float f; std::memcpy(&f,&bits,4); return f; }
static float sigmoid(float x) { return ternip_sigmoid(x); }
static float swish(float x) { return ternip_swish(x); }
static int ternary(const WeightWord& word, unsigned index) {
#ifdef TERNIP_CPU
    unsigned code=(word.bytes[index/4] >> (2*(index%4)))&3;
#else
    unsigned code=word.range(index*2+1,index*2);
#endif
    return code==3 ? -1 : int(code); // exporter forbids reserved code 2
}

#if defined(TERNIP_RTL_TILES)
static void put_i16(WeightWord& word, unsigned j, int16_t value) {
#ifdef TERNIP_CPU
    word.bytes[2*j]=uint16_t(value)&255; word.bytes[2*j+1]=uint16_t(value)>>8;
#else
    word.range(16*j+15,16*j)=uint16_t(value);
#endif
}
static int16_t get_i16(const WeightWord& word, unsigned j) {
#ifdef TERNIP_CPU
    return int16_t(uint16_t(word.bytes[2*j]) | uint16_t(word.bytes[2*j+1])<<8);
#else
    return int16_t(uint16_t(word.range(16*j+15,16*j)));
#endif
}
#ifndef TERNIP_CPU
#define tile_send(word) tile_command.write(word)
#define tile_receive() tile_response.read()
#endif
#endif

extern "C" void ternip_group(const WeightWord* weights, float* state,
    uint32_t* control, const uint32_t* program, const float* parameters,
    float* scratch, uint64_t* counters, uint32_t ops, uint32_t hidden,
    uint32_t layers, uint32_t vocab, uint32_t scratch_stride,
    uint32_t state_stride, uint32_t token_offset, uint32_t quantum,
    uint32_t control_words TERNIP_TILE_PORTS) {
#if defined(TERNIP_RTL_TILES) && !defined(TERNIP_CPU)
#pragma HLS INTERFACE axis port=tile_command
#pragma HLS INTERFACE axis port=tile_response
#endif
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=weights max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=state offset=slave bundle=state
#pragma HLS INTERFACE m_axi port=scratch offset=slave bundle=state
#pragma HLS INTERFACE m_axi port=control offset=slave bundle=control
#pragma HLS INTERFACE m_axi port=program offset=slave bundle=control
#pragma HLS INTERFACE m_axi port=parameters offset=slave bundle=control
#pragma HLS INTERFACE m_axi port=counters offset=slave bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control_regs
#pragma HLS INTERFACE s_axilite port=weights bundle=control_regs
#pragma HLS INTERFACE s_axilite port=state bundle=control_regs
#pragma HLS INTERFACE s_axilite port=control bundle=control_regs
#pragma HLS INTERFACE s_axilite port=program bundle=control_regs
#pragma HLS INTERFACE s_axilite port=parameters bundle=control_regs
#pragma HLS INTERFACE s_axilite port=scratch bundle=control_regs
#pragma HLS INTERFACE s_axilite port=counters bundle=control_regs
#pragma HLS INTERFACE s_axilite port=ops bundle=control_regs
#pragma HLS INTERFACE s_axilite port=hidden bundle=control_regs
#pragma HLS INTERFACE s_axilite port=layers bundle=control_regs
#pragma HLS INTERFACE s_axilite port=vocab bundle=control_regs
#pragma HLS INTERFACE s_axilite port=scratch_stride bundle=control_regs
#pragma HLS INTERFACE s_axilite port=state_stride bundle=control_regs
#pragma HLS INTERFACE s_axilite port=token_offset bundle=control_regs
#pragma HLS INTERFACE s_axilite port=quantum bundle=control_regs
#pragma HLS INTERFACE s_axilite port=control_words bundle=control_regs
    uint64_t counts[8];
    for(unsigned j=0;j<8;++j) counts[j]=counters[j];
    bool active[TERNIP_LANES];
    uint32_t tokens[TERNIP_LANES];
    int8_t q[TERNIP_LANES][TERNIP_MAX_VECTOR];
    float normalized[TERNIP_LANES][TERNIP_MAX_VECTOR];
    float scales[TERNIP_LANES];
#pragma HLS ARRAY_PARTITION variable=active complete
#pragma HLS ARRAY_PARTITION variable=q complete dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=TERNIP_TERNARY_PARALLELISM dim=2
#pragma HLS ARRAY_PARTITION variable=normalized complete dim=1
    if (!hidden || hidden>TERNIP_MAX_VECTOR || state_stride<layers*hidden || token_offset>=scratch_stride || control_words<16*TERNIP_LANES) return;
    for(unsigned lane=0;lane<TERNIP_LANES;++lane) {
        uint32_t* r=control+16*lane;
        if(r[0]==1 && r[12]) {
            for(unsigned j=0;j<state_stride;++j) state[lane*state_stride+j]=0.f;
            r[12]=0;
        }
    }
    for(unsigned tick=0;tick<quantum;++tick) {
        bool any=false;
        for(unsigned lane=0;lane<TERNIP_LANES;++lane) {
            uint32_t* r=control+16*lane;
            active[lane]=false;
            if(r[0]!=1) continue;
            if(r[7]) { r[0]=3; continue; }
            if(r[4]>=r[5]) { r[0]=2; continue; }
            if(!r[2] || r[1]>control_words || r[2]>control_words-r[1] || r[9]>control_words || r[10]>control_words-r[9] || r[11]>r[10]) { r[0]=4;r[13]=1;continue; }
            bool emits=r[3]>=r[2]-1;
            if(emits && r[11]>=r[10]) { counts[4]++;continue; }
            uint32_t token=r[3]<r[2] ? control[r[1]+r[3]] : r[8];
            if(token>=vocab) { r[0]=4;r[13]=2;continue; }
            tokens[lane]=token;active[lane]=true;any=true;counts[2]++;
        }
        counts[3]+=TERNIP_LANES;
        if(!any) continue;
        for(unsigned pc=0;pc<ops;++pc) {
            uint32_t ins[16];
            for(unsigned j=0;j<16;++j) ins[j]=program[pc*16+j];
            unsigned op=ins[0],dst=ins[1],src=ins[2],a=ins[3],b=ins[4],n=ins[5];
            if(op==3) {
                unsigned rows=ins[6],cols=ins[7],padded_cols=ins[14];
                if(cols>TERNIP_MAX_VECTOR || padded_cols%256!=0 || ins[8]%64!=0) {
                    for(unsigned lane=0;lane<TERNIP_LANES;++lane) if(active[lane]) {control[lane*16]=4;control[lane*16+13]=3;active[lane]=false;}
                    break;
                }
                // Full-vector RMS/absmax, outside the weight tile loop.
                for(unsigned lane=0;lane<TERNIP_LANES;++lane) if(active[lane]) {
                    float inv=ternip_rms_inverse(scratch+lane*scratch_stride+src,cols,1e-6f),maxabs=1e-5f;
                    for(unsigned j=0;j<cols;++j) {
#pragma HLS UNROLL factor=TERNIP_VECTOR_PARALLELISM
                        float value=scratch[lane*scratch_stride+src+j]*inv*parameters[ins[10]+j];
                        normalized[lane][j]=value;
                        maxabs=std::fmax(maxabs,std::fabs(value));
                    }
                    float multiplier=127.f/maxabs;
                    scales[lane]=1.f/multiplier;
                    for(unsigned j=0;j<cols;++j) {
#pragma HLS UNROLL factor=TERNIP_VECTOR_PARALLELISM
                        q[lane][j]=ternip_quantized(normalized[lane][j],multiplier);
                    }
                }
                bool select=(pc+1<ops && program[(pc+1)*16]==11);
                unsigned best_id[TERNIP_LANES]={};
                float best_value[TERNIP_LANES];
                for(unsigned lane=0;lane<TERNIP_LANES;++lane) best_value[lane]=-std::numeric_limits<float>::infinity();
#ifdef TERNIP_RTL_TILES
                // Each tile's signed-int16 result is exact: 256*127=32512.
                // Accumulate every column tile in int32 BEFORE applying scales.
                for(unsigned row_base=0;row_base<rows;row_base+=256) {
                    int32_t sums[TERNIP_LANES][256]={};
#pragma HLS ARRAY_PARTITION variable=sums complete dim=1
                    for(unsigned col=0;col<padded_cols;col+=256) {
                        WeightWord header=zero_weight_word();
                        uint16_t mask=0;
                        for(unsigned lane=0;lane<TERNIP_LANES;++lane) if(active[lane]) mask|=uint16_t(1)<<lane;
                        put_i16(header,0,int16_t(mask));tile_send(header);
                        for(unsigned lane=0;lane<TERNIP_LANES;++lane) if(active[lane])
                            for(unsigned j=0;j<256;j+=32) {
                                WeightWord vector=zero_weight_word();
                                for(unsigned k=0;k<32;++k) put_i16(vector,k,col+j+k<cols?q[lane][col+j+k]:0);
                                tile_send(vector);
                            }
                        for(unsigned row=0;row<256;++row) {
                            WeightWord beat=zero_weight_word();
                            if(row_base+row<rows) {
                                beat=weights[ins[8]/64+((row_base+row)*padded_cols+col)/256];
                                counts[0]+=64;
                            }
                            tile_send(beat);
                        }
                        for(unsigned lane=0;lane<TERNIP_LANES;++lane) if(active[lane])
                            for(unsigned row=0;row<256;row+=32) {
                                WeightWord result=tile_receive();
                                for(unsigned j=0;j<32;++j) sums[lane][row+j]+=get_i16(result,j);
                            }
                        WeightWord status=tile_receive();
                        if(get_i16(status,0)!=0) {
                            for(unsigned lane=0;lane<TERNIP_LANES;++lane) if(active[lane]) {
                                control[lane*16]=4;control[lane*16+13]=5;
                            }
                            // A protocol error requires reset, never emit a token.
                            for(unsigned j=0;j<8;++j) counters[j]=counts[j];
                            return;
                        }
                    }
                    for(unsigned row=0;row<256 && row_base+row<rows;++row)
                        for(unsigned lane=0;lane<TERNIP_LANES;++lane) if(active[lane]) {
                            float value=ternip_rescale(sums[lane][row],scales[lane],as_float(ins[12]));
                            if(!select) scratch[lane*scratch_stride+dst+row_base+row]=value;
#ifdef TERNIP_DEBUG_LOGITS
                            if(select) scratch[lane*scratch_stride+dst+row_base+row]=value;
#endif
                            if(value>best_value[lane]) {best_value[lane]=value;best_id[lane]=row_base+row;}
                        }
                }
#else
                for(unsigned row=0;row<rows;++row) {
                    int32_t sums[TERNIP_LANES]={};
#pragma HLS ARRAY_PARTITION variable=sums complete
                    for(unsigned col=0;col<padded_cols;col+=256) {
                        WeightWord beat=weights[ins[8]/64+(row*padded_cols+col)/256];
                        counts[0]+=64;
                        for(unsigned j=0;j<256;++j) {
                            #pragma HLS UNROLL factor=TERNIP_TERNARY_PARALLELISM
                            int w=ternary(beat,j);
                            for(unsigned lane=0;lane<TERNIP_LANES;++lane) {
#pragma HLS UNROLL
                                if(active[lane] && col+j<cols) sums[lane]+=int(q[lane][col+j])*w;
                            }
                        }
                    }
                    for(unsigned lane=0;lane<TERNIP_LANES;++lane) if(active[lane])
                        {
                        float value=ternip_rescale(sums[lane],scales[lane],as_float(ins[12]));
                        if(!select) scratch[lane*scratch_stride+dst+row]=value;
#ifdef TERNIP_DEBUG_LOGITS
                        if(select) scratch[lane*scratch_stride+dst+row]=value;
#endif
                        if(value>best_value[lane]) {best_value[lane]=value;best_id[lane]=row;}
                    }
                }
#endif
                if(select) {
                    unsigned token_dst=program[(pc+1)*16+1];
                    for(unsigned lane=0;lane<TERNIP_LANES;++lane) if(active[lane]) scratch[lane*scratch_stride+token_dst]=float(best_id[lane]);
                    pc++;
                }
            } else for(unsigned lane=0;lane<TERNIP_LANES;++lane) if(active[lane]) {
                float* v=scratch+lane*scratch_stride;
                if(op==1) for(unsigned j=0;j<hidden;++j) v[dst+j]=parameters[ins[9]+tokens[lane]*hidden+j];
                else if(op==2) {
                    float inv=ternip_rms_inverse(v+src,n,as_float(ins[13]));
                    for(unsigned j=0;j<n;++j) v[dst+j]=v[src+j]*inv*parameters[ins[9]+j];
                } else if(op==11) {
                    unsigned best=0;float value=v[src];
                    for(unsigned j=1;j<n;++j) if(v[src+j]>value) {best=j;value=v[src+j];}
                    v[dst]=float(best);
                } else for(unsigned j=0;j<n;++j) {
#pragma HLS UNROLL factor=TERNIP_VECTOR_PARALLELISM
                    float x=v[src+j];
                    if(op==4) v[dst+j]=sigmoid(x);
                    else if(op==5) v[dst+j]=swish(x);
                    else if(op==6) {float lower=parameters[ins[9]+j];v[dst+j]=lower+(1.f-lower)*x;}
                    else if(op==7) {
                        unsigned si=lane*state_stride+ins[11]*hidden+j;
                        float f=v[a+j];state[si]=f*state[si]+(1.f-f)*x;v[dst+j]=state[si];
                    } else if(op==8) v[dst+j]=swish(x)*v[src+n+j];
                    else if(op==9) v[dst+j]=v[a+j]+v[b+j];
                    else if(op==10) v[dst+j]=v[a+j]*v[b+j];
                    else {control[lane*16]=4;control[lane*16+13]=4;active[lane]=false;}
                }
            }
            counts[5]++;
        }
        for(unsigned lane=0;lane<TERNIP_LANES;++lane) if(active[lane]) {
            uint32_t* r=control+16*lane;
            if(r[3]<r[2]) r[3]++;
            if(r[3]<r[2]) continue;
            uint32_t token=uint32_t(scratch[lane*scratch_stride+token_offset]);
            control[r[9]+r[11]]=token;r[11]++;r[4]++;r[8]=token;counts[1]++;
            if(token==r[6] || r[4]>=r[5]) r[0]=2;
        }
    }
    for(unsigned j=0;j<8;++j) counters[j]=counts[j];
}
