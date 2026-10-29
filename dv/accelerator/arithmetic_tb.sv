// SPDX-License-Identifier: BSD-3-Clause
module arithmetic_tb;
    logic signed [7:0] a8,b8,add8,sub8;
    logic signed [15:0] a16,b16,add16,sub16;
    int unsigned seed=32'h1234cafe;
    ternip_add #(.FixedPointPrecision(8)) a0(.a_i(a8),.b_i(b8),.y_o(add8));
    ternip_sub #(.FixedPointPrecision(8)) s0(.a_i(a8),.b_i(b8),.y_o(sub8));
    ternip_add #(.FixedPointPrecision(16)) a1(.a_i(a16),.b_i(b16),.y_o(add16));
    ternip_sub #(.FixedPointPrecision(16)) s1(.a_i(a16),.b_i(b16),.y_o(sub16));
    function automatic int clamp(int x,int lo,int hi);
        return x<lo ? lo : x>hi ? hi : x;
    endfunction
    function automatic int unsigned random_word();
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        return seed;
    endfunction
    initial begin
        for(int a=-128;a<128;a++) for(int b=-128;b<128;b++) begin
            a8=8'(a);b8=8'(b);#1;
            if(int'(add8)!=clamp(a+b,-128,127)) $fatal(1,"int8 add");
            if(int'(sub8)!=clamp(a-b,-128,127)) $fatal(1,"int8 sub");
        end
        for(int n=0;n<20000;n++) begin
            a16=16'(random_word());b16=16'(random_word());#1;
            if(int'(add16)!=clamp(int'(a16)+int'(b16),-32768,32767)) $fatal(1,"int16 add");
            if(int'(sub16)!=clamp(int'(a16)-int'(b16),-32768,32767)) $fatal(1,"int16 sub");
        end
        $display("PASS arithmetic: exhaustive int8 and 20000 random int16 add/sub pairs");
        $finish;
    end
endmodule
