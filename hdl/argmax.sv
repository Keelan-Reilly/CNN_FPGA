// argmax.sv
module argmax #(
    parameter DATA_WIDTH = 16,
    parameter DIM        = 10
)(
    input  logic clk, reset, start,
    input  logic signed [DATA_WIDTH-1:0] vec [0:DIM-1],
    output logic [3:0] idx,          // 0..9
    output logic done
);
    typedef enum logic [1:0] {IDLE, RUN, FINISH} state_t;
    state_t state;
    integer i;
    logic signed [DATA_WIDTH-1:0] bestv;
    logic [3:0] besti;

    always_ff @(posedge clk) begin
        if (reset) begin state<=IDLE; done<=0; idx<=0; i<=0; besti<=0; bestv<='0; end
        else case (state)
            IDLE: begin done<=0; if (start) begin i<=1; besti<=0; bestv<=vec[0]; state<=RUN; end end
            RUN: begin
                if (vec[i] > bestv) begin bestv <= vec[i]; besti <= i[3:0]; end
                if (i == DIM-1) state<=FINISH; else i<=i+1;
            end
            FINISH: begin idx<=besti; done<=1; state<=IDLE; end
        endcase
    end
endmodule