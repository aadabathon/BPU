// Leading-zero count of a 2^Log2W-bit word; cnt_o is meaningful when !all_zero_o.
module bpu_lzc_pow2 #(
  parameter int unsigned Log2W = 1
) (
  input  logic [(1<<Log2W)-1:0] in_i,
  output logic                  all_zero_o,
  output logic [Log2W-1:0]      cnt_o
);

  localparam int unsigned W = 1 << Log2W;

  if (Log2W == 1) begin : g_leaf
    assign all_zero_o = ~|in_i;
    assign cnt_o      = ~in_i[1];
  end else begin : g_node
    logic             zero_hi, zero_lo;
    logic [Log2W-2:0] cnt_hi, cnt_lo;

    bpu_lzc_pow2 #(.Log2W(Log2W - 1)) u_hi (
      .in_i(in_i[W-1:W/2]), .all_zero_o(zero_hi), .cnt_o(cnt_hi)
    );
    bpu_lzc_pow2 #(.Log2W(Log2W - 1)) u_lo (
      .in_i(in_i[W/2-1:0]), .all_zero_o(zero_lo), .cnt_o(cnt_lo)
    );

    assign all_zero_o = zero_hi & zero_lo;
    assign cnt_o      = zero_hi ? {1'b1, cnt_lo} : {1'b0, cnt_hi};
  end

endmodule
