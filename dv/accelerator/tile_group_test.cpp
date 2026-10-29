// SPDX-License-Identifier: BSD-3-Clause
#include "ternip_group.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <cstdint>
static uint32_t rng=0x28226;
static unsigned random_word() {rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static void put(WeightWord& w,int j,int value) {w.bytes[2*j]=value&255;w.bytes[2*j+1]=(uint16_t(value)>>8)&255;}
static int get(const WeightWord& w,int j) {return int16_t(uint16_t(w.bytes[2*j])|(uint16_t(w.bytes[2*j+1])<<8));}
int main() {
    try {
        for(unsigned trial=0;trial<16;++trial) {
            unsigned mask=trial%4;
            WeightWord header={};put(header,0,mask);
            if(trial==15) header.bytes[63]=1; // malformed reserved header bits
            tile_send(header);
            std::array<std::array<int,256>,2> q{},expected{};
            for(unsigned lane=0;lane<2;++lane) if(mask&(1<<lane)) {
                for(unsigned j=0;j<256;j+=32) {
                    WeightWord beat={};
                    for(unsigned k=0;k<32;++k) {
                        int value=trial<4 ? (lane ? -127:127):int(random_word()%255)-127;
                        q[lane][j+k]=value;put(beat,k,value);
                    }
                    tile_send(beat);
                }
            }
            if(mask) for(unsigned r=0;r<256;++r) {
                WeightWord beat={};
                for(unsigned c=0;c<256;++c) {
                    int w=trial<4 ? -1:int(random_word()%3)-1;
                    beat.bytes[c/4]|=uint8_t(w&3)<<(2*(c%4));
                    for(unsigned lane=0;lane<2;++lane) expected[lane][r]+=w*q[lane][c];
                }
                tile_send(beat);
            }
            for(unsigned lane=0;lane<2;++lane) if(mask&(1<<lane))
                for(unsigned r=0;r<256;r+=32) {
                    auto beat=tile_receive();
                    for(unsigned j=0;j<32;++j) if(get(beat,j)!=expected[lane][r+j])
                        throw std::runtime_error("Integer tile mismatch");
                }
            auto status=tile_receive();
            if(get(status,0)!=(trial==15)) throw std::runtime_error("Wrong protocol status");
            // Abort after broadcast has begun; neither partial results nor stale
            // lane acknowledgements may leak into the next transaction.
            WeightWord word={};put(word,0,3);tile_send(word);word={};
            for(unsigned beat=0;beat<16+11;++beat) tile_send(word);
            tile_reset();
        }
        std::cout<<"PASS 16 two-lane tile transactions: active masks, extrema, stalls, malformed header, reset during broadcast\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
