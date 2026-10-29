// SPDX-License-Identifier: BSD-3-Clause
// Free-running AXI4-Stream tile kernel, protocol version 1, fixed D=256.
// Command: one header (low Lanes bits = active mask; all other bits zero),
// then eight 512-bit vector beats per active lane in increasing lane order,
// then 256 row-major packed ternary weight beats shared by every active lane.
// Response: eight signed-int16 result beats per active lane, then one status
// beat (bit 0 = sticky lane error). A zero mask consumes only the header and
// produces only status. Reset abandons the entire transaction on both streams.
// Quantized vector inputs must be in [-127,127]; code 2 is not a ternary weight.
module ternip_tile_group #(
    parameter int Lanes=1,
    parameter int Parallelism=128,
    parameter int VectorParallelism=8
)(
    input logic ap_clk,
    input logic ap_rst_n,
    input logic [511:0] command_TDATA,
    input logic command_TVALID,
    output logic command_TREADY,
    output logic [511:0] response_TDATA,
    output logic response_TVALID,
    input logic response_TREADY
);
    localparam int VB=VectorParallelism*16;
    localparam int WB=Parallelism*2;
    localparam int VS=512/VB;
    localparam int WS=512/WB;
    typedef enum logic [2:0] {HEADER, START, VECTOR, WEIGHT, RESULT, STATUS} phase_t;
    phase_t phase;
    logic [Lanes-1:0] active,started;
    logic [Lanes-1:0] start_ready,vector_ready,weight_ready,result_valid,errors;
    logic [Lanes-1:0] start_valid,vector_valid,weight_valid,result_ready;
    logic [Lanes-1:0][VB-1:0] results;
    logic [WB-1:0] weight_data;
    logic broadcast_ready;
    logic [511:0] result_buffer;
    logic result_full;
    int unsigned lane,chunk,subword,weight_chunk;
    logic protocol_error;
    for(genvar g=0;g<Lanes;g++) begin: lanes
        ternip_tile_lane #(.Tile(256),.Parallelism(Parallelism),.VectorParallelism(VectorParallelism)) core (
            .clk_i(ap_clk),.rst_ni(ap_rst_n),.start_valid_i(start_valid[g]),.start_ready_o(start_ready[g]),
            .vector_valid_i(vector_valid[g]),.vector_ready_o(vector_ready[g]),
            .vector_data_i(command_TDATA[subword*VB+:VB]),
            .weight_valid_i(weight_valid[g]),.weight_ready_o(weight_ready[g]),.weight_data_i(weight_data),
            .result_valid_o(result_valid[g]),.result_ready_i(result_ready[g]),.result_data_o(results[g]),
            .busy_o(),.error_o(errors[g])
        );
    end
    ternip_weight_broadcast #(.Lanes(Lanes),.Width(WB)) broadcast (
        .clk_i(ap_clk),.rst_ni(ap_rst_n),.active_i(active),
        .valid_i(phase==WEIGHT && command_TVALID),.ready_o(broadcast_ready),
        .data_i(command_TDATA[subword*WB+:WB]),.valid_o(weight_valid),.ready_i(weight_ready),.data_o(weight_data)
    );
    always_comb begin
        start_valid='0;vector_valid='0;result_ready='0;
        command_TREADY=0;response_TVALID=0;response_TDATA=result_buffer;
        case(phase)
            HEADER: command_TREADY=1;
            START: start_valid=active & ~started;
            VECTOR: if(lane<Lanes && active[lane]) begin
                vector_valid[lane]=command_TVALID;
                command_TREADY=vector_ready[lane] && subword==VS-1;
            end
            WEIGHT: command_TREADY=broadcast_ready && subword==WS-1;
            RESULT: begin
                response_TVALID=result_full;
                if(lane<Lanes && active[lane]) result_ready[lane]=!result_full;
            end
            STATUS: begin
                response_TVALID=1;response_TDATA='0;
                response_TDATA[0]=protocol_error | (|errors);
            end
            default: begin end
        endcase
    end
    always_ff @(posedge ap_clk) begin
        if(!ap_rst_n) begin
            phase<=HEADER;active<='0;started<='0;lane<=0;chunk<=0;subword<=0;weight_chunk<=0;
            result_buffer<='0;result_full<=0;protocol_error<=0;
        end else case(phase)
            HEADER: if(command_TVALID) begin
                active<=command_TDATA[Lanes-1:0];started<='0;
                protocol_error<=|(command_TDATA >> Lanes);
                phase<=(|command_TDATA[Lanes-1:0])?START:STATUS;
                lane<=0;chunk<=0;subword<=0;weight_chunk<=0;
            end
            START: begin
                started<=started | (start_valid & start_ready);
                if((started | (start_valid & start_ready))==active) phase<=VECTOR;
            end
            VECTOR: begin
                if(lane==Lanes) begin phase<=WEIGHT;subword<=0;end
                else if(!active[lane]) lane<=lane+1;
                else if(vector_valid[lane] && vector_ready[lane]) begin
                    for(int v=0;v<VectorParallelism;v++)
                        if($signed(command_TDATA[subword*VB+v*16+:16]) < -127 ||
                           $signed(command_TDATA[subword*VB+v*16+:16]) > 127) protocol_error<=1;
                    if(subword==VS-1) begin
                        subword<=0;
                        if(chunk==7) begin lane<=lane+1;chunk<=0;end else chunk<=chunk+1;
                    end else subword<=subword+1;
                end
            end
            WEIGHT: if(command_TVALID && broadcast_ready) begin
                for(int w=0;w<Parallelism;w++)
                    if(command_TDATA[subword*WB+w*2+:2]==2'b10) protocol_error<=1;
                if(subword==WS-1) begin
                    subword<=0;
                    if(weight_chunk==255) begin phase<=RESULT;lane<=0;chunk<=0;end
                    else weight_chunk<=weight_chunk+1;
                end else subword<=subword+1;
            end
            RESULT: begin
                if(response_TVALID && response_TREADY) begin
                    result_full<=0;
                    if(chunk==7) begin lane<=lane+1;chunk<=0;end else chunk<=chunk+1;
                end
                if(!result_full) begin
                    if(lane==Lanes) phase<=STATUS;
                    else if(!active[lane]) lane<=lane+1;
                    else if(result_valid[lane] && result_ready[lane]) begin
                        result_buffer[subword*VB+:VB]<=results[lane];
                        if(subword==VS-1) begin subword<=0;result_full<=1;end
                        else subword<=subword+1;
                    end
                end
            end
            STATUS: if(response_TREADY) phase<=HEADER;
            default: phase<=HEADER;
        endcase
    end
    initial begin
        if(Lanes<1 || Lanes>16 || !(Parallelism inside {128,256}) ||
           !(VectorParallelism inside {4,8,16})) $fatal(1,"Illegal tile group configuration");
    end
endmodule
