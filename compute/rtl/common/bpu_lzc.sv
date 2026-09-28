// Leading-zero count. Returns Width when the input is zero.
//
// Built as a balanced tree (depth ~log2(Width) mux levels) rather than a
// priority chain: the input is padded on the right with ones to a power of two
// P > Width, so the padded word is never zero and its leading-zero count is the
// answer, Width included.
module bpu_lzc #(
  parameter int unsigned Width = 32
) (
  input  logic [Width-1:0]           in_i,
  output logic [$clog2(Width+1)-1:0] cnt_o
);

  localparam int unsigned L = $clog2(Width + 1);
  localparam int unsigned P = 1 << L;

  logic [P-1:0] padded;
  logic         unused_all_zero;   // impossible: the pad bits are ones

  assign padded = {in_i, {(P - Width){1'b1}}};

  bpu_lzc_pow2 #(.Log2W(L)) u_tree (
    .in_i(padded), .all_zero_o(unused_all_zero), .cnt_o(cnt_o)
  );

endmodule
