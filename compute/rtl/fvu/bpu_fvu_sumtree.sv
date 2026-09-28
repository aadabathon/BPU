// Pairwise fp32 adder tree over N lanes (N a power of two), adjacent lanes first:
// the canonical order of bpuref.fvu.tree_sum restricted to one word.
// Latency = log2(N) * popcount(AddPipe).
module bpu_fvu_sumtree #(
  parameter int unsigned N       = 4,
  parameter logic [2:0]  AddPipe = 3'b111
) (
  input  logic            clk_i,
  input  logic            rst_ni,
  input  logic [N*32-1:0] in_i,
  output logic [31:0]     sum_o
);

  if (N == 1) begin : g_leaf
    logic unused;
    assign unused = clk_i ^ rst_ni;
    assign sum_o  = in_i;
  end else begin : g_node
    logic [31:0] lo, hi;
    bpu_fvu_sumtree #(.N(N/2), .AddPipe(AddPipe)) u_lo (
      .clk_i, .rst_ni, .in_i(in_i[(N/2)*32-1:0]), .sum_o(lo));
    bpu_fvu_sumtree #(.N(N/2), .AddPipe(AddPipe)) u_hi (
      .clk_i, .rst_ni, .in_i(in_i[N*32-1:(N/2)*32]), .sum_o(hi));
    bpu_fp32_add #(.PipeMask(AddPipe)) u_add (
      .clk_i, .rst_ni, .valid_i(1'b1), .a_i(lo), .b_i(hi), .valid_o(), .y_o(sum_o));
  end

endmodule
