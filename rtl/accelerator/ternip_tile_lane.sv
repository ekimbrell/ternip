// SPDX-License-Identifier: BSD-3-Clause
// A streaming integer tile transaction through the unmodified ternip_core.
// LDV v0 -> IMPORT v0 -> GO -> EXPORT v1 -> SV v1. Host intervention is not
// required. Input order is one full activation vector then a row-major matrix.
// Results are signed int16; the model controller accumulates input tiles in int32.
module ternip_tile_lane #(
    parameter int Tile = 256,
    parameter int Parallelism = 128,
    parameter int VectorParallelism = 8,
    localparam int VectorBits = VectorParallelism*16,
    localparam int WeightBits = Parallelism*2
)(
    input logic clk_i,
    input logic rst_ni,
    input logic start_valid_i,
    output logic start_ready_o,
    input logic vector_valid_i,
    output logic vector_ready_o,
    input logic [VectorBits-1:0] vector_data_i,
    input logic weight_valid_i,
    output logic weight_ready_o,
    input logic [WeightBits-1:0] weight_data_i,
    output logic result_valid_o,
    input logic result_ready_i,
    output logic [VectorBits-1:0] result_data_o,
    output logic busy_o,
    output logic error_o
);
    function automatic ternip_pkg::ternip_cfg_t configuration();
        ternip_pkg::ternip_cfg_t c = ternip_pkg::RegressionCfg;
        c.D = Tile;
        c.TmatmulParallelism = Parallelism;
        c.VectorParallelism = VectorParallelism;
        c.FixedPointExponent = 0;
        c.BatchSize = 1;
        return c;
    endfunction
    localparam ternip_pkg::ternip_cfg_t Cfg = configuration();
    typedef ternip_types#(Cfg)::instruction_t instruction_t;
    typedef enum logic [3:0] {IDLE, LOAD, VECTOR, IMPORT, GO, WEIGHTS, EXPORT, STORE, RESULT} phase_t;
    phase_t phase;
    logic instruction_ready, instruction_valid;
    instruction_t instruction;
    logic ls_stream_valid, ls_write, ls_read_ready, ls_write_valid;
    logic [63:0] ls_address, weight_address;
    logic [31:0] ls_length, weight_length;
    logic weight_stream_valid, weight_read_ready;
    logic [VectorBits-1:0] ls_write_data;
    int unsigned chunks;
    assign start_ready_o = phase == IDLE;
    assign busy_o = phase != IDLE;
    assign vector_ready_o = phase == VECTOR && ls_read_ready;
    assign weight_ready_o = phase == WEIGHTS && weight_read_ready;
    assign result_valid_o = phase == RESULT && ls_write_valid;
    assign result_data_o = ls_write_data;
    always_comb begin
        instruction = '0;
        instruction_valid = 1;
        case (phase)
            LOAD: begin instruction.fu=ternip_pkg::LOADSTORE;instruction.loadstore_op=ternip_pkg::LDV;instruction.v_a=0;end
            IMPORT: begin instruction.fu=ternip_pkg::TMATMUL;instruction.tmatmul_op=ternip_pkg::IMPORT;instruction.v_a=0;end
            GO: begin instruction.fu=ternip_pkg::TMATMUL;instruction.tmatmul_op=ternip_pkg::GO;end
            EXPORT: begin instruction.fu=ternip_pkg::TMATMUL;instruction.tmatmul_op=ternip_pkg::EXPORT;instruction.v_a=1;end
            STORE: begin instruction.fu=ternip_pkg::LOADSTORE;instruction.loadstore_op=ternip_pkg::SV;instruction.v_a=1;end
            default: instruction_valid=0;
        endcase
    end
    always_ff @(posedge clk_i) begin
        if(!rst_ni) begin phase<=IDLE;chunks<=0;error_o<=0;end
        else begin
            if(ls_stream_valid && (ls_address!=0 || ls_length!=Tile*2)) error_o<=1;
            if(weight_stream_valid && (weight_address!=0 || weight_length!=Tile*Tile/4)) error_o<=1;
            case(phase)
                IDLE: if(start_valid_i) phase<=LOAD;
                LOAD: if(instruction_ready) begin phase<=VECTOR;chunks<=0;end
                VECTOR: if(vector_valid_i && vector_ready_o) begin
                    chunks<=chunks+1;
                    if(chunks==Tile/VectorParallelism-1) phase<=IMPORT;
                end
                IMPORT: if(instruction_ready) phase<=GO;
                GO: if(instruction_ready) begin phase<=WEIGHTS;chunks<=0;end
                WEIGHTS: if(weight_valid_i && weight_ready_o) begin
                    chunks<=chunks+1;
                    if(chunks==Tile*Tile/Parallelism-1) phase<=EXPORT;
                end
                EXPORT: if(instruction_ready) phase<=STORE;
                STORE: if(instruction_ready) begin phase<=RESULT;chunks<=0;end
                RESULT: if(result_valid_o && result_ready_i) begin
                    chunks<=chunks+1;
                    if(chunks==Tile/VectorParallelism-1) phase<=IDLE;
                end
                default: begin phase<=IDLE;error_o<=1;end
            endcase
        end
    end
    ternip_core #(.Cfg(Cfg)) core (
        .clk_i, .rst_ni, .instruction_ready_o(instruction_ready),
        .instruction_valid_i(instruction_valid), .instruction_i(instruction),
        .loadstore_ddr_stream_ready_i(1'b1), .loadstore_ddr_stream_valid_o(ls_stream_valid),
        .loadstore_ddr_stream_address_o(ls_address), .loadstore_ddr_stream_write_not_read_o(ls_write),
        .loadstore_ddr_stream_length_o(ls_length), .loadstore_ddr_r_ready_o(ls_read_ready),
        .loadstore_ddr_r_valid_i(vector_valid_i && phase==VECTOR), .loadstore_ddr_r_data_i(vector_data_i),
        .loadstore_ddr_w_ready_i(result_ready_i && phase==RESULT), .loadstore_ddr_w_valid_o(ls_write_valid),
        .loadstore_ddr_w_data_o(ls_write_data), .loadstore_ddr_debug_o(),
        .tmatmul_ddr_stream_ready_i(1'b1), .tmatmul_ddr_stream_valid_o(weight_stream_valid),
        .tmatmul_ddr_stream_address_o(weight_address), .tmatmul_ddr_stream_length_o(weight_length),
        .tmatmul_ddr_r_ready_o(weight_read_ready), .tmatmul_ddr_r_valid_i(weight_valid_i && phase==WEIGHTS),
        .tmatmul_ddr_r_data_i(weight_data_i), .stall_active_o(), .stall_clear_i(1'b0)
    );
    initial begin
        if(!(Tile inside {128,256}) || Tile%VectorParallelism!=0 || (Tile*Tile)%Parallelism!=0)
            $fatal(1,"Unsupported integer tile dimensions");
    end
endmodule
