// Testbench system: bpu_fvu on a bpu_sram_shared, plus one host read port and one
// host write port on the same SRAM. The host loads and reads back memory, and while
// an op runs it injects competing traffic so bank conflicts and response delays are
// exercised.
module bpu_fvu_sys #(
  parameter int unsigned VLanes       = 4,
  parameter logic [2:0]  MulPipe      = 3'b111,
  parameter logic [2:0]  AddPipe      = 3'b111,
  parameter logic [4:0]  SfuPipe      = 5'b11111,
  parameter int unsigned RedFifoDepth = 8,
  parameter int unsigned SfuLanes     = VLanes,
  parameter int unsigned RdPorts      = 1,
  parameter int unsigned NSlot        = 4,
  parameter int unsigned WbDepth      = 8,
  parameter int unsigned AccDepth     = 8,
  parameter int unsigned NBanks       = 4,
  parameter int unsigned BankWords    = 256,
  parameter bit          OutReg       = 1'b0,
  parameter bit          Hash         = 1'b1
) (
  input  logic                    clk_i,
  input  logic                    rst_ni,

  input  logic                    cmd_valid_i,
  output logic                    cmd_ready_o,
  input  logic [4:0]              cmd_op_i,
  input  logic [2:0]              cmd_func_i,
  input  logic [4:0]              cmd_half_log2_i,
  input  logic [15:0]             cmd_rows_i,
  input  logic [15:0]             cmd_cols_i,
  input  logic [31:0]             cmd_d_i, cmd_a_i, cmd_b_i, cmd_c_i, cmd_s_i, cmd_t_i,
  input  logic [31:0]             cmd_ds_i, cmd_as_i, cmd_bs_i, cmd_cs_i, cmd_ss_i, cmd_ts_i,
  output logic                    done_o,
  output logic                    err_o,
  output logic                    busy_o,

  input  logic                    host_rd_valid_i,
  output logic                    host_rd_ready_o,
  input  logic [31-$clog2(VLanes):0] host_rd_addr_i,
  output logic                    host_rd_rvalid_o,
  output logic [VLanes*32-1:0]    host_rd_rdata_o,
  input  logic                    host_wr_valid_i,
  output logic                    host_wr_ready_o,
  input  logic [31-$clog2(VLanes):0] host_wr_addr_i,
  input  logic [VLanes-1:0]       host_wr_mask_i,
  input  logic [VLanes*32-1:0]    host_wr_data_i,

  output logic [31:0]             perf_items_o,
  output logic [31:0]             perf_busy_o,
  output logic [31:0]             perf_stall_rd_o,
  output logic [31:0]             perf_stall_wr_o
);

  localparam int unsigned V   = VLanes;
  localparam int unsigned MAW = 32 - $clog2(V);
  localparam int unsigned NP  = RdPorts;

  logic [NP-1:0]        f_rv, f_rr, f_rrv, f_roob;
  logic [NP*MAW-1:0]    f_ra;
  logic [NP*V*32-1:0]   f_rd;
  logic                 f_wv, f_wr, f_woob;
  logic [MAW-1:0]       f_wa;
  logic [V-1:0]         f_wm;
  logic [V*32-1:0]      f_wd;

  bpu_fvu #(
    .VLanes(V), .MulPipe(MulPipe), .AddPipe(AddPipe), .SfuPipe(SfuPipe),
    .RedFifoDepth(RedFifoDepth), .SfuLanes(SfuLanes), .RdPorts(RdPorts), .NSlot(NSlot),
    .WbDepth(WbDepth), .AccDepth(AccDepth), .EnPerf(1'b1)
  ) u_fvu (
    .clk_i, .rst_ni, .cmd_valid_i, .cmd_ready_o, .cmd_op_i, .cmd_func_i, .cmd_half_log2_i,
    .cmd_rows_i, .cmd_cols_i, .cmd_d_i, .cmd_a_i, .cmd_b_i, .cmd_c_i, .cmd_s_i, .cmd_t_i,
    .cmd_ds_i, .cmd_as_i, .cmd_bs_i, .cmd_cs_i, .cmd_ss_i, .cmd_ts_i, .done_o, .err_o,
    .mrd_valid_o(f_rv), .mrd_ready_i(f_rr), .mrd_addr_o(f_ra), .mrd_rvalid_i(f_rrv),
    .mrd_rdata_i(f_rd), .mrd_oob_i(f_roob),
    .mwr_valid_o(f_wv), .mwr_ready_i(f_wr), .mwr_addr_o(f_wa), .mwr_mask_o(f_wm),
    .mwr_data_o(f_wd), .mwr_oob_i(f_woob),
    .busy_o, .perf_items_o, .perf_busy_o, .perf_stall_rd_o, .perf_stall_wr_o);

  // Read ports: the FVU's, then the host's. Write ports: FVU, host.
  logic [NP:0]          rv, rr, rrv, roob;
  logic [(NP+1)*MAW-1:0] ra;
  logic [(NP+1)*V*32-1:0] rd;
  logic [1:0]           wv, wr, woob;

  assign rv = {host_rd_valid_i, f_rv};
  assign ra = {host_rd_addr_i, f_ra};
  assign f_rr  = rr[NP-1:0];
  assign f_rrv = rrv[NP-1:0];
  assign f_rd  = rd[NP*V*32-1:0];
  assign f_roob = roob[NP-1:0];
  assign host_rd_ready_o  = rr[NP];
  assign host_rd_rvalid_o = rrv[NP];
  assign host_rd_rdata_o  = rd[NP*V*32 +: V*32];
  assign wv = {host_wr_valid_i, f_wv};
  assign f_wr = wr[0];
  assign f_woob = woob[0];
  assign host_wr_ready_o = wr[1];

  logic unused_host_oob;
  assign unused_host_oob = roob[NP] ^ woob[1];

  bpu_sram_shared #(
    .NBanks(NBanks), .BankWords(BankWords), .Lanes(V), .NRd(NP + 1), .NWr(2), .AW(MAW),
    .OutReg(OutReg), .Hash(Hash)
  ) u_sram (
    .clk_i, .rst_ni,
    .rd_valid_i(rv), .rd_ready_o(rr), .rd_addr_i(ra), .rd_rvalid_o(rrv), .rd_rdata_o(rd),
    .rd_oob_o(roob),
    .wr_valid_i(wv), .wr_ready_o(wr), .wr_addr_i({host_wr_addr_i, f_wa}),
    .wr_mask_i({host_wr_mask_i, f_wm}), .wr_data_i({host_wr_data_i, f_wd}), .wr_oob_o(woob),
    .perf_rd_wait_o(), .perf_wr_wait_o());

endmodule
