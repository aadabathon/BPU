// Balanced max tree over N unsigned keys (N a power of two), with an optional
// register after every level so wide trees meet timing next to the lane adder
// tree. Ties need no rule: equal order keys are equal values.
// Latency = log2(N) * RegLevels.
module bpu_fvu_maxtree #(
  parameter int unsigned N         = 4,
  parameter int unsigned W         = 32,
  parameter bit          RegLevels = 1'b1
) (
  input  logic           clk_i,
  input  logic           rst_ni,
  input  logic [N*W-1:0] in_i,
  output logic [W-1:0]   max_o
);

  if (N == 1) begin : g_leaf
    logic unused;
    assign unused = clk_i ^ rst_ni;
    assign max_o  = in_i;
  end else begin : g_node
    logic [W-1:0] lo, hi, m;
    bpu_fvu_maxtree #(.N(N/2), .W(W), .RegLevels(RegLevels)) u_lo (
      .clk_i, .rst_ni, .in_i(in_i[(N/2)*W-1:0]), .max_o(lo));
    bpu_fvu_maxtree #(.N(N/2), .W(W), .RegLevels(RegLevels)) u_hi (
      .clk_i, .rst_ni, .in_i(in_i[N*W-1:(N/2)*W]), .max_o(hi));
    assign m = (hi > lo) ? hi : lo;
    bpu_delay #(.Width(W), .Depth(int'(RegLevels))) u_reg (
      .clk_i, .rst_ni, .d_i(m), .q_o(max_o));
  end

endmodule
