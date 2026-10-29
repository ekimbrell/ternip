// SPDX-License-Identifier: BSD-3-Clause
// Replicated legacy vector cores. Each lane has its own registers, requests,
// scratch addressing and stall status. A scheduler must supply lane-specific
// load/store addresses. Connect tmatmul data through ternip_weight_broadcast
// only after synchronizing the group's matrix-read requests.
// This module preserves the core arithmetic; it is not a complete LM controller.
module ternip_core_lanes #(
    parameter int LanesPerGroup = 1,
    parameter ternip_pkg::ternip_cfg_t Cfg = `TERNIP_CFG
)(
    input  logic                                     clk_i,
    input  logic                                     rst_ni,
    output logic                                     instruction_ready_o [LanesPerGroup],
    input  logic                                     instruction_valid_i [LanesPerGroup],
    input  ternip_types#(Cfg)::instruction_t         instruction_i [LanesPerGroup],
    input  logic                                     loadstore_ddr_stream_ready_i [LanesPerGroup],
    output logic                                     loadstore_ddr_stream_valid_o [LanesPerGroup],
    output ternip_types#(Cfg)::ddr_address_t         loadstore_ddr_stream_address_o [LanesPerGroup],
    output logic                                     loadstore_ddr_stream_write_not_read_o [LanesPerGroup],
    output logic [31:0]                              loadstore_ddr_stream_length_o [LanesPerGroup],
    output logic                                     loadstore_ddr_r_ready_o [LanesPerGroup],
    input  logic                                     loadstore_ddr_r_valid_i [LanesPerGroup],
    input  ternip_types#(Cfg)::vector_chunk_t        loadstore_ddr_r_data_i [LanesPerGroup],
    input  logic                                     loadstore_ddr_w_ready_i [LanesPerGroup],
    output logic                                     loadstore_ddr_w_valid_o [LanesPerGroup],
    output ternip_types#(Cfg)::vector_chunk_t        loadstore_ddr_w_data_o [LanesPerGroup],
    output logic [63:0]                              loadstore_ddr_debug_o [LanesPerGroup],
    input  logic                                     tmatmul_ddr_stream_ready_i [LanesPerGroup],
    output logic                                     tmatmul_ddr_stream_valid_o [LanesPerGroup],
    output ternip_types#(Cfg)::ddr_address_t         tmatmul_ddr_stream_address_o [LanesPerGroup],
    output logic [31:0]                              tmatmul_ddr_stream_length_o [LanesPerGroup],
    output logic                                     tmatmul_ddr_r_ready_o [LanesPerGroup],
    input  logic                                     tmatmul_ddr_r_valid_i [LanesPerGroup],
    input  ternip_types#(Cfg)::tmatmul_stream_data_t tmatmul_ddr_r_data_i [LanesPerGroup],
    output logic                                     stall_active_o [LanesPerGroup],
    input  logic                                     stall_clear_i [LanesPerGroup]
);
    initial begin
        if (Cfg.BatchSize != 1) $fatal(1, "Use LanesPerGroup; core BatchSize must be 1");
        if (LanesPerGroup < 1) $fatal(1, "At least one request lane required");
    end
    for (genvar lane=0; lane<LanesPerGroup; lane++) begin: lanes
        ternip_core #(.Cfg(Cfg)) core (
            .clk_i(clk_i),
            .rst_ni(rst_ni),
            .instruction_ready_o(instruction_ready_o[lane]),
            .instruction_valid_i(instruction_valid_i[lane]),
            .instruction_i(instruction_i[lane]),
            .loadstore_ddr_stream_ready_i(loadstore_ddr_stream_ready_i[lane]),
            .loadstore_ddr_stream_valid_o(loadstore_ddr_stream_valid_o[lane]),
            .loadstore_ddr_stream_address_o(loadstore_ddr_stream_address_o[lane]),
            .loadstore_ddr_stream_write_not_read_o(loadstore_ddr_stream_write_not_read_o[lane]),
            .loadstore_ddr_stream_length_o(loadstore_ddr_stream_length_o[lane]),
            .loadstore_ddr_r_ready_o(loadstore_ddr_r_ready_o[lane]),
            .loadstore_ddr_r_valid_i(loadstore_ddr_r_valid_i[lane]),
            .loadstore_ddr_r_data_i(loadstore_ddr_r_data_i[lane]),
            .loadstore_ddr_w_ready_i(loadstore_ddr_w_ready_i[lane]),
            .loadstore_ddr_w_valid_o(loadstore_ddr_w_valid_o[lane]),
            .loadstore_ddr_w_data_o(loadstore_ddr_w_data_o[lane]),
            .loadstore_ddr_debug_o(loadstore_ddr_debug_o[lane]),
            .tmatmul_ddr_stream_ready_i(tmatmul_ddr_stream_ready_i[lane]),
            .tmatmul_ddr_stream_valid_o(tmatmul_ddr_stream_valid_o[lane]),
            .tmatmul_ddr_stream_address_o(tmatmul_ddr_stream_address_o[lane]),
            .tmatmul_ddr_stream_length_o(tmatmul_ddr_stream_length_o[lane]),
            .tmatmul_ddr_r_ready_o(tmatmul_ddr_r_ready_o[lane]),
            .tmatmul_ddr_r_valid_i(tmatmul_ddr_r_valid_i[lane]),
            .tmatmul_ddr_r_data_i(tmatmul_ddr_r_data_i[lane]),
            .stall_active_o(stall_active_o[lane]),
            .stall_clear_i(stall_clear_i[lane])
        );
    end
endmodule
