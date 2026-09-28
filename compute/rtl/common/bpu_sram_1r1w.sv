// Technology wrapper: one write port, one read port, 1-cycle registered read.
//
// This behavioral body infers BRAM/URAM on the FPGA. The ASIC flow replaces it
// with a macro wrapper (OpenRAM / DFFRAM) behind the same ports. Engine RTL must
// only instantiate memories through wrappers like this one.
//
// Reading and writing the same address in the same cycle returns undefined data;
// callers must not rely on either ordering.
module bpu_sram_1r1w #(
  parameter int unsigned Depth = 64,
  parameter int unsigned Width = 32
) (
  input  logic                     clk_i,

  input  logic                     we_i,
  input  logic [$clog2(Depth)-1:0] waddr_i,
  input  logic [Width-1:0]         wdata_i,

  input  logic                     re_i,
  input  logic [$clog2(Depth)-1:0] raddr_i,
  output logic [Width-1:0]         rdata_o
);

  logic [Width-1:0] mem_q [Depth];

  always_ff @(posedge clk_i) begin
    if (we_i) mem_q[waddr_i] <= wdata_i;
    if (re_i) rdata_o <= mem_q[raddr_i];
  end

endmodule
