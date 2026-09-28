// FP32 vector unit: executes bpuref.fvu ops on a local scratchpad (SPM),
// bit-exact with bpuref.fvu.execute for every VLanes.
//
// SPM: SpmWords words of VLanes fp32 elements; element e lives in word e / VLanes.
// Ops are 2-D (rows x cols) with per-row strided operands (see bpuref/fvu.py):
// vector operands and their strides are multiples of 64 elements, scalars are
// any element. Illegal descriptors are consumed without running and raise the
// sticky err_cmd_o.
//
// Pipeline: the sequencer walks (row, word) items and issues the item's SPM reads
// (one per cycle: row scalar s, group scalar t, a, b, c); two cycles after its
// last read the item enters the lanes; Latency later its result is written back
// (element-wise ops) or enters bpu_fvu_reduce (row reductions). VVECMAT is an
// accumulate  d = (s[r] * a[r,:]) + (r == 0 ? +0 : d)  row after row; consecutive
// rows are spaced so a row never reads an accumulator word before the previous
// row has written it. Ops run one at a time; the external SPM port is usable
// whenever cmd_ready_o is high.
module bpu_fvu #(
  parameter int unsigned VLanes       = 4,         // power of two, 2..64
  parameter int unsigned SpmWords     = 1024,
  parameter logic [2:0]  MulPipe      = 3'b111,
  parameter logic [2:0]  AddPipe      = 3'b111,
  parameter logic [4:0]  SfuPipe      = 5'b11111,
  parameter int unsigned RedFifoDepth = 8,
  parameter bit          EnPerf       = 1'b1
) (
  input  logic                                   clk_i,
  input  logic                                   rst_ni,

  // Command (one op)
  input  logic                                   cmd_valid_i,
  output logic                                   cmd_ready_o,
  input  logic [4:0]                             cmd_op_i,
  input  logic [2:0]                             cmd_func_i,
  input  logic [4:0]                             cmd_half_log2_i,
  input  logic [15:0]                            cmd_rows_i,
  input  logic [15:0]                            cmd_cols_i,
  input  logic [$clog2(SpmWords*VLanes)-1:0]     cmd_d_i, cmd_a_i, cmd_b_i, cmd_c_i, cmd_s_i, cmd_t_i,
  input  logic [$clog2(SpmWords*VLanes)-1:0]     cmd_ds_i, cmd_as_i, cmd_bs_i, cmd_cs_i, cmd_ss_i, cmd_ts_i,

  // External SPM port (only while cmd_ready_o is high)
  input  logic                                   ext_we_i,
  input  logic [$clog2(SpmWords)-1:0]            ext_waddr_i,
  input  logic [VLanes-1:0]                      ext_wmask_i,
  input  logic [VLanes*32-1:0]                   ext_wdata_i,
  input  logic                                   ext_re_i,
  input  logic [$clog2(SpmWords)-1:0]            ext_raddr_i,
  output logic [VLanes*32-1:0]                   ext_rdata_o,

  output logic                                   busy_o,
  input  logic                                   status_clr_i,
  output logic                                   err_cmd_o,
  output logic [31:0]                            perf_items_o,     // items issued
  output logic [31:0]                            perf_cycles_o     // cycles with an op running
);

  import bpu_compute_pkg::*;

  localparam int unsigned V    = VLanes;
  localparam int unsigned LW   = $clog2(V);
  localparam int unsigned AW   = $clog2(SpmWords * V);
  localparam int unsigned WAW  = $clog2(SpmWords);
  localparam int unsigned Lm   = pipe3_latency(MulPipe);
  localparam int unsigned La   = pipe3_latency(AddPipe);
  localparam int unsigned Ls   = int'(SfuPipe[0]) + int'(SfuPipe[1]) + int'(SfuPipe[2])
                               + int'(SfuPipe[3]) + int'(SfuPipe[4]);
  localparam int unsigned Ltot = (Lm + La > Ls) ? Lm + La : Ls;
  localparam int unsigned VmPeriod = Ltot + 6;      // min cycles between VVECMAT row starts
  localparam logic [4:0]  LopAbs = 5'd14;

  // ---------------------------------------------------------------------------
  // Op decode helpers
  // ---------------------------------------------------------------------------
  function automatic logic op_known(input logic [4:0] op);
    op_known = (op <= FvuVsel) || (op >= FvuRsum && op <= FvuRamax) || (op == FvuVvecmat);
  endfunction
  function automatic logic op_red(input logic [4:0] op);
    op_red = (op >= FvuRsum && op <= FvuRamax);
  endfunction
  function automatic logic op_uses_b(input logic [4:0] op);
    op_uses_b = op inside {FvuVadd, FvuVsub, FvuVmul, FvuVaxpy, FvuVmuladd, FvuVsel, FvuRdot};
  endfunction
  function automatic logic op_uses_c(input logic [4:0] op);
    op_uses_c = op inside {FvuVmuladd, FvuVsel};
  endfunction
  function automatic logic op_uses_s(input logic [4:0] op);
    op_uses_s = op inside {FvuVmuls, FvuVadds, FvuVaxpy, FvuVsel, FvuVvecmat};
  endfunction
  function automatic logic [4:0] op_lane(input logic [4:0] op);
    unique case (op)
      FvuVperm, FvuRsum, FvuRmax: op_lane = FvuVcopy;
      FvuRdot:                    op_lane = FvuVmul;
      FvuRamax:                   op_lane = LopAbs;
      FvuVvecmat:                 op_lane = FvuVaxpy;
      default:                    op_lane = op;
    endcase
  endfunction
  // Smallest power of two >= x (x >= 1).
  function automatic logic [16:0] pow2_ceil(input logic [15:0] x);
    pow2_ceil = 17'd1;
    for (int i = 0; i < 17; i++) if (pow2_ceil < 17'(x)) pow2_ceil = pow2_ceil << 1;
  endfunction

  // ---------------------------------------------------------------------------
  // Command accept and legality
  // ---------------------------------------------------------------------------
  typedef enum logic [1:0] {SIdle, SRun, SDrain} state_e;
  state_e st_q;
  logic   pipe_empty;

  logic cmd_fire, cmd_legal, cmd_red, cmd_ew;
  logic [16:0] p2;

  assign cmd_ready_o = (st_q == SIdle);
  assign cmd_fire    = cmd_valid_i && cmd_ready_o;
  assign cmd_red     = op_red(cmd_op_i);
  assign cmd_ew      = !cmd_red;                   // element-wise and VVECMAT write vectors
  assign p2          = pow2_ceil(cmd_cols_i);

  always_comb begin
    cmd_legal = op_known(cmd_op_i) && (cmd_rows_i != '0) && (cmd_cols_i != '0);
    // Vector operands (and vector destinations) are 64-aligned, rows included.
    if ((cmd_a_i[5:0] | cmd_as_i[5:0]) != '0) cmd_legal = 1'b0;
    if (op_uses_b(cmd_op_i) && (cmd_b_i[5:0] | cmd_bs_i[5:0]) != '0) cmd_legal = 1'b0;
    if (op_uses_c(cmd_op_i) && (cmd_c_i[5:0] | cmd_cs_i[5:0]) != '0) cmd_legal = 1'b0;
    if (cmd_ew && cmd_d_i[5:0] != '0) cmd_legal = 1'b0;
    if (cmd_ew && cmd_op_i != FvuVvecmat && cmd_ds_i[5:0] != '0) cmd_legal = 1'b0;
    if (cmd_op_i == FvuVperm && (cmd_cols_i & ((16'd2 << cmd_half_log2_i) - 16'd1)) != '0)
      cmd_legal = 1'b0;
  end

  // Descriptor registers
  logic [4:0]    op_q, lop_q;
  logic [2:0]    func_q;
  logic [15:0]   rows_q, cols_q;
  logic [16:0]   half_q, wreal_q, wrow_q;
  logic [AW-1:0] ds_q, as_q, bs_q, cs_q, ss_q, ts_q;
  logic          red_q, vm_q, max_q, ub_q, uc_q, us_q, ut_q;

  // ---------------------------------------------------------------------------
  // Sequencer
  // ---------------------------------------------------------------------------
  logic [15:0]   r_q;
  logic [16:0]   w_q;
  logic [AW-1:0] rd_q, ra_q, rb_q, rc_q, rs_q, rt_q;      // row bases
  logic [4:0]    issued_q;                                 // {c, b, a, t, s}
  logic [$clog2(RedFifoDepth+1)-1:0] cred_q;
  logic [$clog2(VmPeriod+1)-1:0]     rowt_q;
  logic          seq_done_q;

  logic [AW-1:0] eoff;                 // element offset of this word within the row
  logic          pad, last_w, last_r, row_start;
  logic [4:0]    need, remain, pick;
  logic          gate_ok, issue, launch;
  logic [AW-1:0] a_el, b_el, c_el, t_el, rd_el;

  assign eoff      = AW'(w_q) << LW;
  assign pad       = red_q && (w_q >= wreal_q);
  assign last_w    = (w_q == wrow_q - 1'b1);
  assign last_r    = (r_q == rows_q - 1'b1);
  assign row_start = (w_q == '0);

  assign need[0] = us_q && row_start;                                  // s
  assign need[1] = ut_q && (eoff[5:0] == '0);                          // t (new 64-group)
  assign need[2] = !pad;                                               // a (every op reads a)
  assign need[3] = (ub_q && !pad) || (vm_q && r_q != '0);              // b (VVECMAT: accumulator)
  assign need[4] = uc_q && !pad;                                       // c
  assign remain  = need & ~issued_q;
  assign pick    = remain & (~remain + 5'd1);                          // lowest set bit

  // Starting a new item: reductions need a FIFO credit; VVECMAT rows are spaced.
  assign gate_ok = (issued_q != '0)
                || ((!red_q || cred_q != '0)
                    && !(vm_q && row_start && r_q != '0 && rowt_q < ($bits(rowt_q))'(VmPeriod)));
  assign issue   = (st_q == SRun) && !seq_done_q && gate_ok && (remain != '0);
  assign launch  = (st_q == SRun) && !seq_done_q && gate_ok && ((remain & ~pick) == '0);

  assign a_el  = ra_q + ((op_q == FvuVperm && half_q >= 17'(V)) ? (eoff ^ AW'(half_q)) : eoff);
  assign b_el  = rb_q + eoff;
  assign c_el  = rc_q + eoff;
  assign t_el  = rt_q + (eoff >> 6);
  assign rd_el = rd_q + eoff;

  // SPM read request for the picked operand
  logic [AW-1:0] rq_el;
  always_comb begin
    unique case (1'b1)
      pick[0]: rq_el = rs_q;
      pick[1]: rq_el = t_el;
      pick[2]: rq_el = a_el;
      pick[3]: rq_el = b_el;
      default: rq_el = c_el;
    endcase
  end

  logic [4:0]    ret_slot_q;
  logic [LW-1:0] ret_lane_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      st_q       <= SIdle;
      issued_q   <= '0;
      seq_done_q <= 1'b0;
      ret_slot_q <= '0;
      r_q        <= '0;
      w_q        <= '0;
      rowt_q     <= '0;
    end else begin
      ret_slot_q <= issue ? pick : 5'd0;
      if (rowt_q != '1) rowt_q <= rowt_q + 1'b1;
      unique case (st_q)
        SIdle: begin
          if (cmd_fire && cmd_legal) begin
            st_q       <= SRun;
            seq_done_q <= 1'b0;
            issued_q   <= '0;
            r_q        <= '0;
            w_q        <= '0;
            rowt_q     <= '0;
          end
        end
        SRun: begin
          if (issue) issued_q <= issued_q | pick;
          if (issue && issued_q == '0 && row_start) rowt_q <= '0;   // a row's first read
          if (launch) begin
            issued_q <= '0;
            if (!last_w) begin
              w_q <= w_q + 1'b1;
            end else begin
              w_q <= '0;
              if (!last_r) r_q <= r_q + 1'b1;
              else         seq_done_q <= 1'b1;
            end
            if (last_w && last_r) st_q <= SDrain;
          end
        end
        SDrain: ;
        default: st_q <= SIdle;
      endcase
      if (st_q == SDrain && pipe_empty) st_q <= SIdle;
    end
  end

  // Row base registers advance by their strides at each row end.
  always_ff @(posedge clk_i) begin
    if (cmd_fire) begin
      op_q    <= cmd_op_i;
      lop_q   <= op_lane(cmd_op_i);
      func_q  <= cmd_func_i;
      rows_q  <= cmd_rows_i;
      cols_q  <= cmd_cols_i;
      half_q  <= 17'd1 << cmd_half_log2_i;
      wreal_q <= 17'((32'(cmd_cols_i) + V - 1) >> LW);
      wrow_q  <= cmd_red ? 17'(((p2 < 17'd64) ? 17'd64 : p2) >> LW) : 17'((32'(cmd_cols_i) + V - 1) >> LW);
      red_q   <= cmd_red;
      vm_q    <= (cmd_op_i == FvuVvecmat);
      max_q   <= (cmd_op_i == FvuRmax) || (cmd_op_i == FvuRamax);
      ub_q    <= op_uses_b(cmd_op_i);
      uc_q    <= op_uses_c(cmd_op_i);
      us_q    <= op_uses_s(cmd_op_i);
      ut_q    <= (cmd_op_i == FvuVmulg);
      ds_q    <= (cmd_op_i == FvuVvecmat) ? '0 : cmd_ds_i;
      as_q    <= cmd_as_i;
      bs_q    <= (cmd_op_i == FvuVvecmat) ? '0 : cmd_bs_i;
      cs_q    <= cmd_cs_i;
      ss_q    <= cmd_ss_i;
      ts_q    <= cmd_ts_i;
      rd_q    <= cmd_d_i;
      ra_q    <= cmd_a_i;
      rb_q    <= (cmd_op_i == FvuVvecmat) ? cmd_d_i : cmd_b_i;
      rc_q    <= cmd_c_i;
      rs_q    <= cmd_s_i;
      rt_q    <= cmd_t_i;
    end else if (launch && last_w) begin
      rd_q <= rd_q + ds_q;
      ra_q <= ra_q + as_q;
      rb_q <= rb_q + bs_q;
      rc_q <= rc_q + cs_q;
      rs_q <= rs_q + ss_q;
      rt_q <= rt_q + ts_q;
    end
  end

  // Reduction FIFO credits
  logic red_pop;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) cred_q <= ($bits(cred_q))'(RedFifoDepth);
    else         cred_q <= cred_q - ($bits(cred_q))'(launch && red_q) + ($bits(cred_q))'(red_pop);
  end

  // ---------------------------------------------------------------------------
  // SPM
  // ---------------------------------------------------------------------------
  logic               spm_we, spm_re;
  logic [WAW-1:0]     spm_waddr, spm_raddr;
  logic [V-1:0]       spm_wmask;
  logic [V*32-1:0]    spm_wdata, spm_rdata;

  bpu_sram_1r1w_be #(.Depth(SpmWords), .Lanes(V), .LaneW(32)) u_spm (
    .clk_i, .we_i(spm_we), .waddr_i(spm_waddr), .wmask_i(spm_wmask), .wdata_i(spm_wdata),
    .re_i(spm_re), .raddr_i(spm_raddr), .rdata_o(spm_rdata));

  assign spm_re      = issue || (cmd_ready_o && ext_re_i);
  assign spm_raddr   = issue ? WAW'(rq_el >> LW) : ext_raddr_i;
  assign ext_rdata_o = spm_rdata;

  always_ff @(posedge clk_i) ret_lane_q <= rq_el[LW-1:0];

  // Staging registers (written the cycle after each read)
  logic [31:0]   st_s, st_t;
  logic [V*32-1:0] st_a, st_b, st_c;
  always_ff @(posedge clk_i) begin
    if (ret_slot_q[0]) st_s <= spm_rdata[ret_lane_q*32 +: 32];
    if (ret_slot_q[1]) st_t <= spm_rdata[ret_lane_q*32 +: 32];
    if (ret_slot_q[2]) st_a <= spm_rdata;
    if (ret_slot_q[3]) st_b <= spm_rdata;
    if (ret_slot_q[4]) st_c <= spm_rdata;
  end

  // ---------------------------------------------------------------------------
  // Item metadata: launch -> L1 -> L2 (emit into the lanes)
  // ---------------------------------------------------------------------------
  logic [V-1:0] mask;
  for (genvar l = 0; l < V; l++) begin : g_mask
    assign mask[l] = !pad && ((32'(eoff) + l) < 32'(cols_q));
  end

  localparam int unsigned MW = 1 + 1 + 1 + 1 + 1 + V + AW;   // zb, red, max, first, last, mask, dest
  logic          l1_v, l2_v;
  logic [MW-1:0] l1_m, l2_m, lnow_m;
  assign lnow_m = {vm_q && r_q == '0, red_q, max_q, row_start, last_w, mask,
                   red_q ? rd_q : rd_el};

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      l1_v <= 1'b0;
      l2_v <= 1'b0;
    end else begin
      l1_v <= launch;
      l2_v <= l1_v;
    end
  end
  always_ff @(posedge clk_i) begin
    l1_m <= lnow_m;
    l2_m <= l1_m;
  end

  logic e_zb;
  assign e_zb = l2_m[MW-1];            // VVECMAT first row: accumulate onto +0

  // Operand a, with VPERM's in-word lane swap (partner distance < VLanes).
  logic [V*32-1:0] e_a;
  for (genvar l = 0; l < V; l++) begin : g_perm
    always_comb begin
      e_a[l*32 +: 32] = st_a[l*32 +: 32];
      if (op_q == FvuVperm && half_q < 17'(V))
        e_a[l*32 +: 32] = st_a[(l ^ int'(half_q[LW-1:0]))*32 +: 32];
    end
  end

  // ---------------------------------------------------------------------------
  // Lanes
  // ---------------------------------------------------------------------------
  logic [V*32-1:0] res;
  for (genvar l = 0; l < V; l++) begin : g_lane
    bpu_fvu_lane #(.MulPipe(MulPipe), .AddPipe(AddPipe), .SfuPipe(SfuPipe)) u_lane (
      .clk_i, .rst_ni, .valid_i(l2_v), .lop_i(lop_q), .func_i(func_q), .zero_b_i(e_zb),
      .a_i(e_a[l*32 +: 32]), .b_i(st_b[l*32 +: 32]), .c_i(st_c[l*32 +: 32]),
      .s_i(st_s), .t_i(st_t), .res_o(res[l*32 +: 32]));
  end

  logic          o_v;
  logic          o_red, o_max, o_first, o_last;
  logic [V-1:0]  o_mask;
  logic [AW-1:0] o_dest;
  logic          unused_o_zb;

  bpu_delay #(.Width(1), .Depth(Ltot), .Reset(1'b1)) u_ov (
    .clk_i, .rst_ni, .d_i(l2_v), .q_o(o_v));
  bpu_delay #(.Width(MW), .Depth(Ltot)) u_om (
    .clk_i, .rst_ni, .d_i(l2_m), .q_o({unused_o_zb, o_red, o_max, o_first, o_last, o_mask, o_dest}));

  // ---------------------------------------------------------------------------
  // Reductions
  // ---------------------------------------------------------------------------
  logic          rw_v, red_busy;
  logic [AW-1:0] rw_addr;
  logic [31:0]   rw_data;

  bpu_fvu_reduce #(.Lanes(V), .AW(AW), .AddPipe(AddPipe), .FifoDepth(RedFifoDepth)) u_red (
    .clk_i, .rst_ni,
    .in_valid_i(o_v && o_red), .in_max_i(o_max), .in_first_i(o_first), .in_last_i(o_last),
    .in_dest_i(o_dest), .in_mask_i(o_mask), .in_vals_i(res),
    .pop_o(red_pop), .wr_valid_o(rw_v), .wr_addr_o(rw_addr), .wr_data_o(rw_data),
    .busy_o(red_busy));

  // ---------------------------------------------------------------------------
  // Write port: element-wise results, reduction results, or the external port
  // ---------------------------------------------------------------------------
  always_comb begin
    spm_we    = 1'b0;
    spm_waddr = ext_waddr_i;
    spm_wmask = ext_wmask_i;
    spm_wdata = ext_wdata_i;
    if (o_v && !o_red) begin
      spm_we    = 1'b1;
      spm_waddr = WAW'(o_dest >> LW);
      spm_wmask = o_mask;
      spm_wdata = res;
    end else if (rw_v) begin
      spm_we    = 1'b1;
      spm_waddr = WAW'(rw_addr >> LW);
      spm_wmask = V'(1) << rw_addr[LW-1:0];
      spm_wdata = {V{rw_data}};
    end else if (cmd_ready_o && ext_we_i) begin
      spm_we    = 1'b1;
    end
  end

  // ---------------------------------------------------------------------------
  // Completion, status, counters
  // ---------------------------------------------------------------------------
  logic [$clog2(Ltot+8)-1:0] inflight_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) inflight_q <= '0;
    else         inflight_q <= inflight_q + ($bits(inflight_q))'(launch) - ($bits(inflight_q))'(o_v);
  end
  assign pipe_empty = (inflight_q == '0) && !red_busy && !rw_v;
  assign busy_o     = (st_q != SIdle);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)                        err_cmd_o <= 1'b0;
    else if (status_clr_i)              err_cmd_o <= 1'b0;
    else if (cmd_fire && !cmd_legal)    err_cmd_o <= 1'b1;
  end

  if (EnPerf) begin : g_perf
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        perf_items_o  <= '0;
        perf_cycles_o <= '0;
      end else if (status_clr_i) begin
        perf_items_o  <= '0;
        perf_cycles_o <= '0;
      end else begin
        perf_items_o  <= perf_items_o + 32'(launch);
        perf_cycles_o <= perf_cycles_o + 32'(st_q != SIdle);
      end
    end
  end else begin : g_no_perf
    assign perf_items_o  = '0;
    assign perf_cycles_o = '0;
  end


`ifndef SYNTHESIS
  /* verilator lint_off SYNCASYNCNET */
  initial begin
    if (V < 2 || V > 64 || (V & (V - 1)) != 0)
      $fatal(1, "bpu_fvu: VLanes=%0d must be a power of two in [2, 64]", V);
  end
  always @(posedge clk_i) begin
    if (rst_ni && !cmd_ready_o && (ext_we_i || ext_re_i))
      $error("bpu_fvu: external SPM access while an op is running");
  end
  /* verilator lint_on SYNCASYNCNET */
`endif

endmodule
