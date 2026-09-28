// Fixed-latency delay line. Depth == 0 is a plain wire, so parents can
// align paths whose latencies are parameter-dependent without special cases.
// Set Reset for control bits (valid); leave it off for datapath bits so they
// map onto SRLs / reset-free flops.
module bpu_delay #(
  parameter int unsigned Width = 1,
  parameter int unsigned Depth = 1,
  parameter bit          Reset = 1'b0
) (
  input  logic             clk_i,
  input  logic             rst_ni,
  input  logic [Width-1:0] d_i,
  output logic [Width-1:0] q_o
);

  if (Depth == 0 || !Reset) begin : g_no_rst
    logic unused_rst_n;
    assign unused_rst_n = rst_ni;
  end

  if (Depth == 0) begin : g_wire
    logic unused_clk;
    assign unused_clk = clk_i;
    assign q_o = d_i;
  end else begin : g_regs
    // Packed so synthesis never infers a RAM for a shift register.
    logic [Depth-1:0][Width-1:0] stage_q;

    if (Reset) begin : g_rst
      always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
          for (int unsigned i = 0; i < Depth; i++) stage_q[i] <= '0;
        end else begin
          stage_q[0] <= d_i;
          for (int unsigned i = 1; i < Depth; i++) stage_q[i] <= stage_q[i-1];
        end
      end
    end else begin : g_norst
      always_ff @(posedge clk_i) begin
        stage_q[0] <= d_i;
        for (int unsigned i = 1; i < Depth; i++) stage_q[i] <= stage_q[i-1];
      end
    end

    assign q_o = stage_q[Depth-1];
  end

endmodule
