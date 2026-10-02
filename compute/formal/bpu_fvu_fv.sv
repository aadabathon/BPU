// Formal harness for bpu_fvu on a bpu_sram_shared, with one more SRAM client (any
// read and write requests, AXI-stable) competing for the banks. Any sequence of op
// descriptors (shapes bounded to keep the search small). The arithmetic (lanes,
// adder trees, merge adder) and the SRAM contents are cut out in the .sby script,
// so the properties in bpu_fvu, bpu_fvu_reduce and bpu_sram_shared (`ifdef FORMAL)
// must hold for any data and any memory timing the arbiter produces.
module bpu_fvu_fv #(
  parameter int unsigned VLanes   = 2,
  parameter int unsigned RdPorts  = 1,
  parameter int unsigned AccDepth = 2,
  parameter bit          SmallAddr = 1'b0
) (
  input logic                         clk_i,
  input logic                         rst_ni,
  input logic                         cmd_valid_i,
  input logic [4:0]                   cmd_op_i,
  input logic [2:0]                   cmd_func_i,
  input logic [4:0]                   cmd_half_log2_i,
  input logic [15:0]                  cmd_rows_i,
  input logic [15:0]                  cmd_cols_i,
  input logic [31:0]                  cmd_d_i, cmd_a_i, cmd_b_i, cmd_c_i, cmd_s_i, cmd_t_i,
  input logic [31:0]                  cmd_ds_i, cmd_as_i, cmd_bs_i, cmd_cs_i, cmd_ss_i, cmd_ts_i,
  input logic                         host_rv_i,
  input logic [31-$clog2(VLanes):0]   host_ra_i,
  input logic                         host_wv_i,
  input logic [31-$clog2(VLanes):0]   host_wa_i
);

  localparam int unsigned V   = VLanes;
  localparam int unsigned MAW = 32 - $clog2(V);
  localparam int unsigned NP  = RdPorts;

  logic cmd_ready, done, err;
  logic [NP-1:0]      f_rv, f_rr, f_rrv, f_roob;
  logic [NP*MAW-1:0]  f_ra;
  logic [NP*V*32-1:0] f_rd;
  logic               f_wv, f_wr, f_woob;
  logic [MAW-1:0]     f_wa;
  logic [V-1:0]       f_wm;
  logic [V*32-1:0]    f_wd;

  bpu_fvu #(
    .VLanes(V), .MulPipe(3'b001), .AddPipe(3'b001), .SfuPipe(5'b00001), .RedFifoDepth(2),
    .RdPorts(NP), .NSlot(2), .WbDepth(2), .AccDepth(AccDepth), .EnPerf(1'b0)
  ) dut (
    .clk_i, .rst_ni, .cmd_valid_i, .cmd_ready_o(cmd_ready), .cmd_op_i, .cmd_func_i,
    .cmd_half_log2_i, .cmd_rows_i, .cmd_cols_i, .cmd_d_i, .cmd_a_i, .cmd_b_i, .cmd_c_i,
    .cmd_s_i, .cmd_t_i, .cmd_ds_i, .cmd_as_i, .cmd_bs_i, .cmd_cs_i, .cmd_ss_i, .cmd_ts_i,
    .done_o(done), .err_o(err),
    .mrd_valid_o(f_rv), .mrd_ready_i(f_rr), .mrd_addr_o(f_ra), .mrd_rvalid_i(f_rrv),
    .mrd_rdata_i(f_rd), .mrd_oob_i(f_roob),
    .mwr_valid_o(f_wv), .mwr_ready_i(f_wr), .mwr_addr_o(f_wa), .mwr_mask_o(f_wm),
    .mwr_data_o(f_wd), .mwr_oob_i(f_woob),
    .busy_o(), .perf_items_o(), .perf_busy_o(), .perf_stall_rd_o(), .perf_stall_wr_o());

  logic [NP:0]            rr, rrv, roob;
  logic [(NP+1)*V*32-1:0] rd;
  logic [1:0]             wr, woob;
  assign f_rr = rr[NP-1:0];
  assign f_rrv = rrv[NP-1:0];
  assign f_rd = rd[NP*V*32-1:0];
  assign f_roob = roob[NP-1:0];
  assign f_wr = wr[0];
  assign f_woob = woob[0];

  bpu_sram_shared #(
    .NBanks(2), .BankWords(2), .Lanes(V), .NRd(NP + 1), .NWr(2), .AW(MAW), .OutReg(1'b0), .Hash(1'b1)
  ) u_sram (
    .clk_i, .rst_ni,
    .rd_valid_i({host_rv_i, f_rv}), .rd_ready_o(rr), .rd_addr_i({host_ra_i, f_ra}),
    .rd_rvalid_o(rrv), .rd_rdata_o(rd), .rd_oob_o(roob),
    .wr_valid_i({host_wv_i, f_wv}), .wr_ready_o(wr), .wr_addr_i({host_wa_i, f_wa}),
    .wr_mask_i({{V{1'b1}}, f_wm}), .wr_data_i({{(V*32){1'b0}}, f_wd}), .wr_oob_o(woob),
    .perf_rd_wait_o(), .perf_wr_wait_o());

  // ---------------------------------------------------------------------------
  // Environment
  // ---------------------------------------------------------------------------
  logic init_q = 1'b1, rst_n_q;
  logic             hrv_q, hrr_q, hwv_q, hwr_q;
  logic [MAW-1:0]   hra_q, hwa_q;
  always_ff @(posedge clk_i) begin
    init_q  <= 1'b0;
    rst_n_q <= rst_ni;
    hrv_q <= host_rv_i;  hrr_q <= rr[NP];  hra_q <= host_ra_i;
    hwv_q <= host_wv_i;  hwr_q <= wr[1];   hwa_q <= host_wa_i;
  end

  always_comb begin
    if (init_q) assume (!rst_ni);
    if (!init_q && rst_n_q) assume (rst_ni);
    // Small shapes keep the state space tractable; alignment rules are the DUT's job.
    assume (cmd_rows_i <= 16'd3 && cmd_cols_i <= 16'd6);
    // Small addresses (still past the 8-element SRAM, so out-of-range paths are covered).
    if (SmallAddr)
      assume ((cmd_d_i | cmd_a_i | cmd_b_i | cmd_c_i | cmd_s_i | cmd_t_i | cmd_ds_i | cmd_as_i
               | cmd_bs_i | cmd_cs_i | cmd_ss_i | cmd_ts_i) < 32'd256);
    // The other client keeps a request stable until it is accepted.
    if (!init_q && rst_n_q && hrv_q && !hrr_q) assume (host_rv_i && host_ra_i == hra_q);
    if (!init_q && rst_n_q && hwv_q && !hwr_q) assume (host_wv_i && host_wa_i == hwa_q);
  end

  // ---------------------------------------------------------------------------
  // Checks: every FVU request is served within the round-robin bound
  // ---------------------------------------------------------------------------
  logic [NP-1:0][2:0] rwait_q;
  logic [2:0]         wwait_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rwait_q <= '0;
      wwait_q <= '0;
    end else begin
      for (int p = 0; p < NP; p++) rwait_q[p] <= (f_rv[p] && !f_rr[p]) ? rwait_q[p] + 1'b1 : '0;
      wwait_q <= (f_wv && !f_wr) ? wwait_q + 1'b1 : '0;
    end
  end

  logic done_one_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)                 done_one_q <= 1'b0;
    else if (done && !err)       done_one_q <= 1'b1;
  end

  always_comb begin
    if (rst_ni) begin
      for (int p = 0; p < NP; p++) assert (rwait_q[p] <= 3'(NP));
      assert (wwait_q <= 3'd1);
      cover (done_one_q && cmd_ready);                 // a legal op completes
      cover (dut.u_red.wr_valid_o);                     // a reduction produces its row result
      cover (done && !err && dut.vm_q && dut.fwd_q);    // a forwarded VVECMAT completes
    end
  end

endmodule
