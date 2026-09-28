// Technology wrapper: 1 write + 1 read port, 1-cycle registered read, with a
// write-enable per Lanes-wide segment (e.g. one per fp32 element of a word).
// Same contract as bpu_sram_1r1w; FPGA BRAM/URAM infer segment enables.
module bpu_sram_1r1w_be #(
  parameter int unsigned Depth = 64,
  parameter int unsigned Lanes = 4,
  parameter int unsigned LaneW = 32
) (
  input  logic                     clk_i,

  input  logic                     we_i,
  input  logic [$clog2(Depth)-1:0] waddr_i,
  input  logic [Lanes-1:0]         wmask_i,
  input  logic [Lanes*LaneW-1:0]   wdata_i,

  input  logic                     re_i,
  input  logic [$clog2(Depth)-1:0] raddr_i,
  output logic [Lanes*LaneW-1:0]   rdata_o
);

  logic [Lanes*LaneW-1:0] mem_q [Depth];

  always_ff @(posedge clk_i) begin
    if (we_i) begin
      for (int unsigned l = 0; l < Lanes; l++) begin
        if (wmask_i[l]) mem_q[waddr_i][l*LaneW +: LaneW] <= wdata_i[l*LaneW +: LaneW];
      end
    end
    if (re_i) rdata_o <= mem_q[raddr_i];
  end

endmodule
