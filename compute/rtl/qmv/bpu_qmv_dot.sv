// Integer dot product of one weight beat with one activation word.
//
// A beat holds Lanes 4-bit nibbles (Lanes*4 bits):
//   W4: every lane is one signed int4 weight; lane i uses x element i.
//   W8: lane pair (2j, 2j+1) is one int8 weight = signed(hi) * 16 + unsigned(lo).
//       Even lanes multiply the unsigned low nibble, odd lanes multiply the
//       signed high nibble and shift left by 4. Both use x element j of the
//       selected half (x_hi_i) of the activation word.
// So one 5x8-bit signed multiplier per lane serves both formats, and a W8 beat
// carries Lanes/2 weights. The result is exact.
//
// Latency = ProdReg + add_tree_latency(Lanes, TreeRegEvery).
module bpu_qmv_dot #(
  parameter int unsigned Lanes        = 64,
  parameter bit          ProdReg      = 1'b1,
  parameter int unsigned TreeRegEvery = 2
) (
  input  logic                              clk_i,
  input  logic                              w8_i,     // bpu_compute_pkg::WfmtW8
  input  logic                              x_hi_i,   // W8: use the upper half of x_i
  input  logic [Lanes*4-1:0]                w_i,
  input  logic [Lanes*8-1:0]                x_i,      // Lanes signed int8 activations
  output logic signed [17+$clog2(Lanes)-1:0] sum_o
);

  // |int8 * nibble| <= 2^11; the W8 high-nibble term is shifted by 4, so 17 bits.
  localparam int unsigned TermW = 17;

  logic [Lanes*TermW-1:0] terms_d, terms_q;

  for (genvar i = 0; i < Lanes; i++) begin : g_lane
    localparam bit IsOdd = (i % 2) == 1;

    logic [3:0]              nib;
    logic signed [4:0]       wv;
    logic signed [7:0]       xv;
    logic signed [12:0]      prod;
    logic signed [TermW-1:0] term;

    assign nib = w_i[4*i +: 4];
    // W8 low nibbles are unsigned; everything else is a signed nibble.
    assign wv  = (w8_i && !IsOdd) ? $signed({1'b0, nib}) : $signed({nib[3], nib});
    assign xv  = !w8_i  ? x_i[8*i +: 8]
               : x_hi_i ? x_i[8*(Lanes/2 + i/2) +: 8]
                        : x_i[8*(i/2) +: 8];
    assign prod = wv * xv;
    assign term = (w8_i && IsOdd) ? (TermW'(prod) <<< 4) : TermW'(prod);

    assign terms_d[TermW*i +: TermW] = term;
  end

  if (ProdReg) begin : g_prod_reg
    always_ff @(posedge clk_i) terms_q <= terms_d;
  end else begin : g_prod_comb
    assign terms_q = terms_d;
  end

  bpu_add_tree #(.N(Lanes), .InW(TermW), .RegEvery(TreeRegEvery)) u_tree (
    .clk_i, .in_i(terms_q), .sum_o
  );

endmodule
