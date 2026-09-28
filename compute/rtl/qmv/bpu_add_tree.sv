// Balanced signed adder tree over N inputs (N a power of two).
// Each level grows by one bit, so it never overflows: OutW = InW + log2(N).
//
// Node outputs at levels that are multiples of RegEvery are registered
// (level = log2 of the node's input count), so every path through the tree
// has the same latency: add_tree_latency(N, RegEvery). RegEvery = 0 builds a
// purely combinational tree. Integer addition is exact, so tree shape and
// pipelining never change results.
module bpu_add_tree #(
  parameter int unsigned N        = 8,
  parameter int unsigned InW      = 16,
  parameter int unsigned RegEvery = 2
) (
  input  logic                                clk_i,
  input  logic [N*InW-1:0]                    in_i,   // N packed two's-complement values
  output logic signed [InW+$clog2(N)-1:0]     sum_o
);

  localparam int unsigned Levels = $clog2(N);
  localparam int unsigned OutW   = InW + Levels;

  if (N == 1) begin : g_leaf
    logic unused_clk;
    assign unused_clk = clk_i;
    assign sum_o = $signed(in_i);
  end else begin : g_node
    logic signed [OutW-2:0] sum_lo, sum_hi;
    logic signed [OutW-1:0] sum_d;

    bpu_add_tree #(.N(N/2), .InW(InW), .RegEvery(RegEvery)) u_lo (
      .clk_i, .in_i(in_i[(N/2)*InW-1:0]), .sum_o(sum_lo)
    );
    bpu_add_tree #(.N(N/2), .InW(InW), .RegEvery(RegEvery)) u_hi (
      .clk_i, .in_i(in_i[N*InW-1:(N/2)*InW]), .sum_o(sum_hi)
    );

    assign sum_d = OutW'(sum_lo) + OutW'(sum_hi);

    if (RegEvery != 0 && (Levels % RegEvery) == 0) begin : g_reg
      always_ff @(posedge clk_i) sum_o <= sum_d;
    end else begin : g_comb
      assign sum_o = sum_d;
    end
  end

endmodule
