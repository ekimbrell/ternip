// SPDX-License-Identifier: BSD-3-Clause
module broadcast_tb;
    localparam int Lanes=4;
    logic clk=0;
    always #5 clk=~clk;
    logic rst_n=0,valid,ready;
    logic [Lanes-1:0] active,out_valid,out_ready;
    logic [31:0] data,out_data;
    logic [Lanes-1:0] expected_pending;
    logic [31:0] expected_data;
    int accepted=0,delivered=0;
    int unsigned seed=32'h739218;
    ternip_weight_broadcast #(.Lanes(Lanes),.Width(32)) dut (
        .clk_i(clk),.rst_ni(rst_n),.active_i(active),.valid_i(valid),.ready_o(ready),
        .data_i(data),.valid_o(out_valid),.ready_i(out_ready),.data_o(out_data)
    );
    function automatic int unsigned random_word();
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        return seed;
    endfunction
    initial begin
        valid=0;active=0;out_ready=0;data=0;expected_pending=0;expected_data=0;
        repeat(2) @(negedge clk);
        for(int cycle=0;cycle<10000;cycle++) begin
            rst_n=(cycle%193)!=0;
            valid=random_word()%4!=0;
            active=random_word();out_ready=random_word();data=random_word();
            #1;
            if(!rst_n) begin expected_pending=0;expected_data=0;end
            if(out_valid!==expected_pending) $fatal(1,"Duplicate/lost delivery at cycle %0d",cycle);
            if((|expected_pending) && out_data!==expected_data) $fatal(1,"Stalled payload changed");
            if(ready!==~(|(expected_pending & ~out_ready))) $fatal(1,"Invalid upstream acknowledgement");
            @(posedge clk);
            if(rst_n) begin
                delivered += $countones(expected_pending & out_ready);
                if(valid && ready) begin
                    expected_pending=active;expected_data=data;accepted++;
                end else expected_pending &= ~out_ready;
            end
            @(negedge clk);
        end
        $display("PASS broadcast: %0d accepted beats, %0d lane deliveries, randomized stalls and reset",accepted,delivered);
        $finish;
    end
endmodule
