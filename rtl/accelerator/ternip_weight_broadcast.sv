// SPDX-License-Identifier: BSD-3-Clause
// One buffered beat, delivered exactly once to each lane active at acceptance.
// active_i is sampled only on an upstream handshake. A stalled lane keeps its
// own pending bit and the common payload stable. Reset flushes an in-flight beat.
module ternip_weight_broadcast #(
    parameter int Lanes = 1,
    parameter int Width = 512
)(
    input logic clk_i,
    input logic rst_ni,
    input logic [Lanes-1:0] active_i,
    input logic valid_i,
    output logic ready_o,
    input logic [Width-1:0] data_i,
    output logic [Lanes-1:0] valid_o,
    input logic [Lanes-1:0] ready_i,
    output logic [Width-1:0] data_o
);
    logic [Lanes-1:0] pending;
    logic [Lanes-1:0] remaining;
    assign remaining = pending & ~ready_i;
    assign ready_o = ~(|remaining);
    assign valid_o = pending;
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            pending <= '0;
            data_o <= '0;
        end else if (ready_o && valid_i) begin
            pending <= active_i;
            data_o <= data_i;
        end else begin
            pending <= remaining;
        end
    end
    initial begin
        if (Lanes < 1 || Width < 1) $fatal(1, "Invalid broadcast dimensions");
    end
endmodule
