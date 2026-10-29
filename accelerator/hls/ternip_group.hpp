// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
#ifndef TERNIP_LANES
#define TERNIP_LANES 1
#endif
#ifndef TERNIP_TERNARY_PARALLELISM
#define TERNIP_TERNARY_PARALLELISM 128
#endif
#ifndef TERNIP_VECTOR_PARALLELISM
#define TERNIP_VECTOR_PARALLELISM 4
#endif
#ifndef TERNIP_MAX_VECTOR
#define TERNIP_MAX_VECTOR 8192
#endif
#ifdef TERNIP_CPU
struct WeightWord { uint8_t bytes[64]; };
#else
#include <ap_int.h>
typedef ap_uint<512> WeightWord;
#endif
#if defined(TERNIP_RTL_TILES) && !defined(TERNIP_CPU)
#include <hls_stream.h>
#define TERNIP_TILE_PORTS , hls::stream<WeightWord>& tile_command, hls::stream<WeightWord>& tile_response
#else
#define TERNIP_TILE_PORTS
#endif
#if defined(TERNIP_RTL_TILES) && defined(TERNIP_CPU)
// Supplied by the RTL co-simulation adapter, not by the model controller.
void tile_send(const WeightWord& word);
WeightWord tile_receive();
void tile_reset();
#endif
inline WeightWord zero_weight_word() {
#ifdef TERNIP_CPU
    return WeightWord{};
#else
    return WeightWord(0);
#endif
}
// Requests are 16 uint32 words per lane; see device.py's versioned layout.
// Buffers remain resident across calls. Host may change requests only BETWEEN
// bounded kernel invocations. No CPU feedback is needed inside a token quantum.
extern "C" void ternip_group(const WeightWord* weights, float* state,
    uint32_t* control, const uint32_t* program, const float* parameters,
    float* scratch, uint64_t* counters, uint32_t ops, uint32_t hidden,
    uint32_t layers, uint32_t vocab, uint32_t scratch_stride,
    uint32_t state_stride, uint32_t token_offset, uint32_t quantum,
    uint32_t control_words TERNIP_TILE_PORTS);
