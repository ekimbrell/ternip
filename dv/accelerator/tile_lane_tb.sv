// SPDX-License-Identifier: BSD-3-Clause
module tile_lane_tb;
    parameter int Tile=128;
    parameter int Parallelism=128;
    localparam int VP=8;
    logic clk=0;
    always #5 clk=~clk;
    logic rst_n=0,start_valid=0,start_ready,vector_valid=0,vector_ready;
    logic weight_valid=0,weight_ready,result_valid,result_ready=0,busy,error;
    logic [VP*16-1:0] vector_data,result_data;
    logic [Parallelism*2-1:0] weight_data;
    int values[Tile],weights[Tile][Tile],expected[Tile];
    int unsigned seed=32'h1235abcd;
    int delivered;
    ternip_tile_lane #(.Tile(Tile),.Parallelism(Parallelism),.VectorParallelism(VP)) dut (
        .clk_i(clk),.rst_ni(rst_n),.start_valid_i(start_valid),.start_ready_o(start_ready),
        .vector_valid_i(vector_valid),.vector_ready_o(vector_ready),.vector_data_i(vector_data),
        .weight_valid_i(weight_valid),.weight_ready_o(weight_ready),.weight_data_i(weight_data),
        .result_valid_o(result_valid),.result_ready_i(result_ready),.result_data_o(result_data),
        .busy_o(busy),.error_o(error)
    );
    function automatic int unsigned random_word();
        seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;
    endfunction
    task automatic launch();
        @(negedge clk);start_valid=1;
        do @(posedge clk); while(!start_ready);
        @(negedge clk);start_valid=0;
    endtask
    task automatic vector_beat(input int offset);
        @(negedge clk);
        for(int j=0;j<VP;j++) vector_data[j*16+:16]=16'(values[offset+j]);
        vector_valid=1;
        do @(posedge clk);while(!vector_ready);
        @(negedge clk);vector_valid=0;
    endtask
    task automatic weight_beat(input int offset);
        @(negedge clk);
        for(int j=0;j<Parallelism;j++) weight_data[j*2+:2]=2'(weights[(offset+j)/Tile][(offset+j)%Tile]);
        weight_valid=1;
        do @(posedge clk);while(!weight_ready);
        @(negedge clk);weight_valid=0;
    endtask
    initial begin
        repeat(3) @(negedge clk);rst_n=1;
        for(int trial=0;trial<8;trial++) begin
            for(int j=0;j<Tile;j++) values[j]=(trial==0)?127:(trial==1)?-127:int'(random_word()%255)-127;
            for(int r=0;r<Tile;r++) begin
                expected[r]=0;
                for(int c=0;c<Tile;c++) begin
                    weights[r][c]=(trial<2)?-1:int'(random_word()%3)-1;
                    expected[r]+=weights[r][c]*values[c];
                end
            end
            launch();
            for(int j=0;j<Tile;j+=VP) vector_beat(j);
            for(int j=0;j<Tile*Tile;j+=Parallelism) begin
                if(random_word()%5==0) repeat(3) @(negedge clk);
                weight_beat(j);
            end
            delivered=0;
            while(delivered<Tile) begin
                @(negedge clk);result_ready=(random_word()%3)!=0;
                @(posedge clk);
                if(result_valid && result_ready) begin
                    for(int j=0;j<VP;j++)
                        if(int'($signed(result_data[j*16+:16]))!=expected[delivered+j])
                            $fatal(1,"tile trial=%0d row=%0d got=%0d expected=%0d",trial,delivered+j,$signed(result_data[j*16+:16]),expected[delivered+j]);
                    delivered+=VP;
                end
                if(error) $fatal(1,"Tile stream contract error");
            end
            @(negedge clk);result_ready=0;
            // Reset an interrupted vector load, then start a clean transaction.
            launch();vector_beat(0);
            @(negedge clk);rst_n=0;
            repeat(3) @(negedge clk);rst_n=1;
        end
        $display("PASS real ternip_core: 8 integer matmul tiles, input/output stalls, interrupted resets, D=%0d P=%0d",Tile,Parallelism);
        $finish;
    end
    initial begin #2000000; $fatal(1,"Tile test timed out");end
endmodule
