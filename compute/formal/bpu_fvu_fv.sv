// Formal harness for bpu_fvu: any sequence of op descriptors (shapes bounded to
// keep the search small), external-port traffic only while idle. The arithmetic
// (lanes, adder trees, SPM data) is cut out in the .sby script, so the properties
// in bpu_fvu / bpu_fvu_reduce (`ifdef FORMAL) must hold for any data.
module bpu_fvu_fv #(
  parameter int unsigned VLanes   = 2,
  parameter int unsigned SpmWords = 16
) (
  input logic                                clk_i,
  input logic                                rst_ni,
  input logic                                cmd_valid_i,
  input logic [4:0]                          cmd_op_i,
  input logic [2:0]                          cmd_func_i,
  input logic [4:0]                          cmd_half_log2_i,
  input logic [15:0]                         cmd_rows_i,
  input logic [15:0]                         cmd_cols_i,
  input logic [$clog2(SpmWords*VLanes)-1:0]  cmd_d_i, cmd_a_i, cmd_b_i, cmd_c_i, cmd_s_i, cmd_t_i,
  input logic [$clog2(SpmWords*VLanes)-1:0]  cmd_ds_i, cmd_as_i, cmd_bs_i, cmd_cs_i, cmd_ss_i, cmd_ts_i,
  input logic                                ext_we_i,
  input logic [$clog2(SpmWords)-1:0]         ext_waddr_i,
  input logic [VLanes-1:0]                   ext_wmask_i,
  input logic [VLanes*32-1:0]                ext_wdata_i,
  input logic                                ext_re_i,
  input logic [$clog2(SpmWords)-1:0]         ext_raddr_i,
  input logic                                status_clr_i
);

  logic cmd_ready_o, busy_o, err_cmd_o;

  bpu_fvu #(
    .VLanes(VLanes), .SpmWords(SpmWords), .MulPipe(3'b001), .AddPipe(3'b001),
    .SfuPipe(5'b00001), .RedFifoDepth(2), .EnPerf(1'b0)
  ) dut (
    .clk_i, .rst_ni, .cmd_valid_i, .cmd_ready_o, .cmd_op_i, .cmd_func_i, .cmd_half_log2_i,
    .cmd_rows_i, .cmd_cols_i, .cmd_d_i, .cmd_a_i, .cmd_b_i, .cmd_c_i, .cmd_s_i, .cmd_t_i,
    .cmd_ds_i, .cmd_as_i, .cmd_bs_i, .cmd_cs_i, .cmd_ss_i, .cmd_ts_i,
    .ext_we_i, .ext_waddr_i, .ext_wmask_i, .ext_wdata_i, .ext_re_i, .ext_raddr_i, .ext_rdata_o(),
    .busy_o, .status_clr_i, .err_cmd_o, .perf_items_o(), .perf_cycles_o()
  );

  logic init_q = 1'b1, rst_n_q, done_one_q;
  always_ff @(posedge clk_i) begin
    init_q  <= 1'b0;
    rst_n_q <= rst_ni;
  end

  always_comb begin
    if (init_q) assume (!rst_ni);
    if (!init_q && rst_n_q) assume (rst_ni);
    // Small shapes keep the state space tractable; alignment rules are the DUT's job.
    assume (cmd_rows_i <= 16'd3 && cmd_cols_i <= 16'd6);
    // The external port is only used while the FVU is idle (documented rule).
    if (!cmd_ready_o) assume (!ext_we_i && !ext_re_i);
  end

  // Reachability: a legal op runs to completion.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)                        done_one_q <= 1'b0;
    else if (busy_o && !done_one_q)     done_one_q <= 1'b1;
  end
  always_comb begin
    if (rst_ni) begin
      cover (done_one_q && cmd_ready_o && !err_cmd_o);
      cover (dut.u_red.wr_valid_o);            // a reduction produced its row result
    end
  end

endmodule
