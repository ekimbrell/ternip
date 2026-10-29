// SPDX-License-Identifier: BSD-3-Clause
// Cycle-accurate adapter for the *actual* RTL tile group. The same resident
// model controller and request buffers are used by cpu_runner and this runner.
#include "ternip_group.hpp"
#include "Vternip_tile_group.h"
#include "verilated.h"
#include <stdexcept>
#include <iostream>

namespace {
struct Transport {
    VerilatedContext context;
    Vternip_tile_group top{&context};
    uint32_t rng=0x281abcd;
    uint64_t cycles=0,transactions=0;
    unsigned random() {rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
    void evaluate() {top.ap_clk=0;top.eval();}
    void advance() {
        uint64_t until=context.time()+5;
        while(top.eventsPending() && top.nextTimeSlot()<=until) {
            context.time(top.nextTimeSlot());top.eval();
        }
        context.time(until);
    }
    void tick() {
        top.ap_clk=1;top.eval();advance();
        top.ap_clk=0;top.eval();advance();++cycles;
        if(cycles%1000000==0) std::cerr<<"RTL progress cycles="<<cycles<<"\n";
        if(context.gotFinish()) throw std::runtime_error("Unexpected RTL simulation termination");
    }
    Transport() {
        top.ap_rst_n=0;top.command_TVALID=0;top.response_TREADY=0;
        for(int i=0;i<5;++i) tick();
        top.ap_rst_n=1;tick();
    }
    ~Transport() {top.final();std::cout<<"RTL tile transport cycles="<<cycles<<" response_beats="<<transactions<<"\n";}
};
Transport& transport() {static Transport t;return t;}
}
void tile_send(const WeightWord& word) {
    auto& t=transport();
    for(unsigned i=0,n=t.random()%3;i<n;++i) t.tick();
    for(unsigned j=0;j<16;++j) {
        uint32_t value=0;
        for(unsigned k=0;k<4;++k) value|=uint32_t(word.bytes[4*j+k])<<(8*k);
        t.top.command_TDATA[j]=value;
    }
    t.top.command_TVALID=1;t.evaluate();
    for(unsigned wait=0;wait<100000;++wait) {
        bool ready=t.top.command_TREADY;t.tick();
        if(ready) {t.top.command_TVALID=0;t.evaluate();return;}
    }
    throw std::runtime_error("RTL command timeout");
}
WeightWord tile_receive() {
    auto& t=transport();
    for(unsigned wait=0;wait<100000;++wait) {
        t.top.response_TREADY=(t.random()%4)!=0;t.evaluate();
        bool accepted=t.top.response_TVALID && t.top.response_TREADY;
        WeightWord result={};
        if(accepted) for(unsigned j=0;j<16;++j) for(unsigned k=0;k<4;++k)
            result.bytes[4*j+k]=(t.top.response_TDATA[j]>>(8*k))&255;
        t.tick();
        if(accepted) {t.top.response_TREADY=0;t.evaluate();++t.transactions;return result;}
    }
    throw std::runtime_error("RTL response timeout");
}

void tile_reset() {
    auto& t=transport();
    t.top.command_TVALID=0;t.top.response_TREADY=0;t.top.ap_rst_n=0;
    for(int i=0;i<5;++i) t.tick();
    t.top.ap_rst_n=1;t.tick();
}
