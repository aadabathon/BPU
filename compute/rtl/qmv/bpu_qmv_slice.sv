// QMV slice: quantized matrix-vector product y = W x for one stripe of rows.
// Semantics are defined in compute/docs/numerics.md ("QMV"); the reference is
// bpuref.qmv.qmv_ref. Outputs are bit-identical for every legal parameter set.
//
// Usage per operation:
//   1. While cmd_ready_o is high, write the activation codes (x_*) and the
//      per-group activation scales (xs_*) into the local buffers.
//   2. Issue a command {wfmt, ngroups = K/64, nrowblk = N/RowInterleave}.
//   3. Stream weight beats on w_* and one bf16 weight scale per (row, group)
//      on ws_*, in "layout L0" order (docs/engine-ops.md):
//        for rb < nrowblk: for g < ngroups: for r < R: for c < chunks:
//          beat(row = rb*R + r, group g, chunk c)
//      The scale for (row, g) is consumed with that row-group's last chunk.
//   4. Results leave y_* in row order, one fp32 per row.
//
// Throughput: one weight beat per cycle when streams and output keep up.
// Datapath: accept -> x read -> int dot -> group sum -> int2fp | scale mul
//           -> p = isum * scale -> fp32 accumulate (R rows interleaved) -> FIFO.
module bpu_qmv_slice #(
  parameter int unsigned Lanes         = 64,     // int4 MAC lanes: power of two, 2..64
  parameter int unsigned RowInterleave = 4,      // rows in flight (R)
  parameter int unsigned MaxK          = 6144,   // largest K; multiple of 64, >= 128
  parameter int unsigned RowBlkW       = 20,     // width of the row-block count
  parameter int unsigned OutFifoDepth  = 2 * RowInterleave,
  parameter bit          ProdReg       = 1'b1,
  parameter int unsigned TreeRegEvery  = 2,
  parameter bit          I2fReg        = 1'b1,
  parameter logic [2:0]  MulPipe       = 3'b111,
  parameter logic [2:0]  AddPipe       = 3'b111
) (
  input  logic                                clk_i,
  input  logic                                rst_ni,

  // Activation buffer: Lanes int8 codes per word, element k in word k / Lanes.
  input  logic                                x_we_i,
  input  logic [$clog2(MaxK/Lanes)-1:0]       x_waddr_i,
  input  logic [Lanes*8-1:0]                  x_wdata_i,
  // Activation scales: one bf16 per group of 64 elements.
  input  logic                                xs_we_i,
  input  logic [$clog2(MaxK/64)-1:0]          xs_waddr_i,
  input  logic [15:0]                         xs_wdata_i,

  // Command
  input  logic                                cmd_valid_i,
  output logic                                cmd_ready_o,
  input  logic                                cmd_wfmt_i,     // bpu_compute_pkg::Wfmt*
  input  logic [$clog2(MaxK/64+1)-1:0]        cmd_ngroups_i,  // K / 64, >= 1
  input  logic [RowBlkW-1:0]                  cmd_nrowblk_i,  // N / RowInterleave, >= 1

  // Weight code beats
  input  logic                                w_valid_i,
  output logic                                w_ready_o,
  input  logic [Lanes*4-1:0]                  w_data_i,

  // Weight scales (bf16), one per (row, group)
  input  logic                                ws_valid_i,
  output logic                                ws_ready_o,
  input  logic [15:0]                         ws_data_i,

  // Results (fp32), in row order
  output logic                                y_valid_o,
  input  logic                                y_ready_i,
  output logic [31:0]                         y_data_o,

  output logic                                busy_o
);

  import bpu_compute_pkg::*;

  localparam int unsigned G      = QmvGroup;
  localparam int unsigned R      = RowInterleave;
  localparam int unsigned Cpg4   = G / Lanes;                 // chunks per group in W4
  localparam int unsigned XAW    = $clog2(MaxK / Lanes);
  localparam int unsigned XSAW   = $clog2(MaxK / G);
  localparam int unsigned GW     = $clog2(MaxK / G + 1);
  localparam int unsigned CW     = $clog2(2 * Cpg4);          // W8 has 2 * Cpg4 chunks
  localparam int unsigned RW     = (R > 1) ? $clog2(R) : 1;
  localparam int unsigned CrW    = $clog2(OutFifoDepth + 1);
  localparam int unsigned SumW   = 17 + $clog2(Lanes);
  localparam int unsigned DotLat = int'(ProdReg) + add_tree_latency(Lanes, TreeRegEvery);
  localparam int unsigned MulLat = pipe3_latency(MulPipe);
  localparam int unsigned AddLat = pipe3_latency(AddPipe);
  localparam int unsigned I2fLat = int'(I2fReg);
  localparam int unsigned ScLat  = (MulLat > I2fLat) ? MulLat : I2fLat;

  // ---------------------------------------------------------------------------
  // Accept stage: command, loop counters, stream handshakes, output credits
  // ---------------------------------------------------------------------------
  logic               run_q, w8_q;
  logic [GW-1:0]      ngroups_q, g_q;
  logic [RowBlkW-1:0] nrowblk_q, rb_q;
  logic [CW-1:0]      c_q, cpg_m1;
  logic [RW-1:0]      r_q;
  logic [CrW-1:0]     credits_q;

  logic last_c, last_r, last_g, last_rb, first_beat_rb;
  logic credit_ok, scale_ok, accept, cmd_fire, y_pop;

  assign cpg_m1        = w8_q ? CW'(2 * Cpg4 - 1) : CW'(Cpg4 - 1);
  assign last_c        = (c_q == cpg_m1);
  assign last_r        = (r_q == RW'(R - 1));
  assign last_g        = (g_q == ngroups_q - 1'b1);
  assign last_rb       = (rb_q == nrowblk_q - 1'b1);
  assign first_beat_rb = (g_q == '0) && (r_q == '0) && (c_q == '0);

  // A row block starts only when R output slots are free, so the non-stalling
  // pipeline behind this point can always write its results.
  assign credit_ok = !first_beat_rb || (credits_q >= CrW'(R));
  // A row-group's last chunk needs its weight scale in the same cycle.
  assign scale_ok  = !last_c || ws_valid_i;

  assign w_ready_o   = run_q && credit_ok && scale_ok;
  assign ws_ready_o  = run_q && credit_ok && last_c && w_valid_i;
  assign accept      = w_valid_i && w_ready_o;
  assign cmd_ready_o = !run_q;
  assign cmd_fire    = cmd_valid_i && cmd_ready_o;
  assign y_pop       = y_valid_o && y_ready_i;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      run_q     <= 1'b0;
      w8_q      <= 1'b0;
      ngroups_q <= '0;
      nrowblk_q <= '0;
      c_q       <= '0;
      r_q       <= '0;
      g_q       <= '0;
      rb_q      <= '0;
    end else if (cmd_fire) begin
      run_q     <= 1'b1;
      w8_q      <= cmd_wfmt_i;
      ngroups_q <= cmd_ngroups_i;
      nrowblk_q <= cmd_nrowblk_i;
      c_q       <= '0;
      r_q       <= '0;
      g_q       <= '0;
      rb_q      <= '0;
    end else if (accept) begin
      if (!last_c) begin
        c_q <= c_q + 1'b1;
      end else begin
        c_q <= '0;
        if (!last_r) begin
          r_q <= r_q + 1'b1;
        end else begin
          r_q <= '0;
          if (!last_g) begin
            g_q <= g_q + 1'b1;
          end else begin
            g_q <= '0;
            if (!last_rb) begin
              rb_q <= rb_q + 1'b1;
            end else begin
              rb_q  <= '0;
              run_q <= 1'b0;
            end
          end
        end
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      credits_q <= CrW'(OutFifoDepth);
    end else begin
      credits_q <= credits_q
                 - ((accept && first_beat_rb) ? CrW'(R) : CrW'(0))
                 + (y_pop ? CrW'(1) : CrW'(0));
    end
  end

  // Idle only when no row is reserved, in flight, or waiting in the FIFO.
  assign busy_o = run_q || (credits_q != CrW'(OutFifoDepth));

  // ---------------------------------------------------------------------------
  // Local buffers (read at accept; data arrives in stage B)
  // ---------------------------------------------------------------------------
  logic [XAW-1:0]     x_raddr;
  logic [CW-1:0]      c_word;
  logic [Lanes*8-1:0] b_x;
  logic [15:0]        b_xs;

  // Word index = g * Cpg4 + chunk (W8 chunks cover half a word each).
  assign c_word  = w8_q ? (c_q >> 1) : c_q;
  assign x_raddr = XAW'(g_q) * XAW'(Cpg4) + XAW'(c_word);

  bpu_sram_1r1w #(.Depth(MaxK / Lanes), .Width(Lanes * 8)) u_xbuf (
    .clk_i,
    .we_i(x_we_i), .waddr_i(x_waddr_i), .wdata_i(x_wdata_i),
    .re_i(accept), .raddr_i(x_raddr), .rdata_o(b_x)
  );

  bpu_sram_1r1w #(.Depth(MaxK / G), .Width(16)) u_xsbuf (
    .clk_i,
    .we_i(xs_we_i), .waddr_i(xs_waddr_i), .wdata_i(xs_wdata_i),
    .re_i(accept), .raddr_i(XSAW'(g_q)), .rdata_o(b_xs)
  );

  // ---------------------------------------------------------------------------
  // Stage B: registered beat + tags, integer dot product
  // ---------------------------------------------------------------------------
  logic               b_valid_q;
  logic [Lanes*4-1:0] b_w_q;
  logic               b_w8_q, b_xhi_q, b_first_c_q, b_last_c_q, b_first_g_q, b_last_g_q;
  logic [RW-1:0]      b_r_q;
  logic [15:0]        b_ws_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) b_valid_q <= 1'b0;
    else         b_valid_q <= accept;
  end

  always_ff @(posedge clk_i) begin
    if (accept) begin
      b_w_q       <= w_data_i;
      b_w8_q      <= w8_q;
      b_xhi_q     <= c_q[0];
      b_first_c_q <= (c_q == '0);
      b_last_c_q  <= last_c;
      b_first_g_q <= (g_q == '0);
      b_last_g_q  <= last_g;
      b_r_q       <= r_q;
      b_ws_q      <= ws_data_i;
    end
  end

  logic signed [SumW-1:0] d_sum;

  bpu_qmv_dot #(.Lanes(Lanes), .ProdReg(ProdReg), .TreeRegEvery(TreeRegEvery)) u_dot (
    .clk_i, .w8_i(b_w8_q), .x_hi_i(b_xhi_q), .w_i(b_w_q), .x_i(b_x), .sum_o(d_sum)
  );

  // Tags ride alongside the dot product.
  localparam int unsigned DTagW = 4 + RW + 16 + 16;

  logic             d_valid, d_first_c, d_last_c, d_first_g, d_last_g;
  logic [RW-1:0]    d_r;
  logic [15:0]      d_ws, d_xs;

  bpu_delay #(.Width(1), .Depth(DotLat), .Reset(1'b1)) u_dv (
    .clk_i, .rst_ni, .d_i(b_valid_q), .q_o(d_valid)
  );
  bpu_delay #(.Width(DTagW), .Depth(DotLat)) u_dtag (
    .clk_i, .rst_ni,
    .d_i({b_first_c_q, b_last_c_q, b_first_g_q, b_last_g_q, b_r_q, b_ws_q, b_xs}),
    .q_o({d_first_c,   d_last_c,   d_first_g,   d_last_g,   d_r,   d_ws,   d_xs})
  );

  // ---------------------------------------------------------------------------
  // Stage E: sum the chunks of one row-group (exact integer)
  // ---------------------------------------------------------------------------
  logic signed [QmvIsumW-1:0] d_sum_ext, gacc_d, gacc_q, e_isum_q;
  logic                       e_valid_q, e_first_g_q, e_last_g_q;
  logic [RW-1:0]              e_r_q;
  logic [15:0]                e_ws_q, e_xs_q;

  // |beat sum| <= |group sum| <= 2^20, so resizing to QmvIsumW is lossless.
  assign d_sum_ext = QmvIsumW'(d_sum);
  if (SumW > QmvIsumW) begin : g_sum_trunc
    logic unused_sum_msbs;
    assign unused_sum_msbs = ^d_sum[SumW-1:QmvIsumW];
  end
  assign gacc_d    = d_first_c ? d_sum_ext : gacc_q + d_sum_ext;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) e_valid_q <= 1'b0;
    else         e_valid_q <= d_valid && d_last_c;
  end

  always_ff @(posedge clk_i) begin
    if (d_valid) gacc_q <= gacc_d;
    if (d_valid && d_last_c) begin
      e_isum_q    <= gacc_d;
      e_first_g_q <= d_first_g;
      e_last_g_q  <= d_last_g;
      e_r_q       <= d_r;
      e_ws_q      <= d_ws;
      e_xs_q      <= d_xs;
    end
  end

  // ---------------------------------------------------------------------------
  // Stage F: isum -> fp32 in parallel with scale = f32(ws) * f32(xs)
  // ---------------------------------------------------------------------------
  logic [31:0] e_isum_f, e_sc, f_isum, f_sc;
  logic        f_valid, f_first_g, f_last_g;
  logic [RW-1:0] f_r;

  bpu_int2fp32 #(.InW(QmvIsumW), .Reg(I2fReg)) u_i2f (
    .clk_i, .rst_ni, .valid_i(e_valid_q), .x_i(e_isum_q), .valid_o(), .y_o(e_isum_f)
  );

  // bf16 * bf16 is exact in fp32 unless it overflows or underflows.
  bpu_fp32_mul #(.PipeMask(MulPipe)) u_scale_mul (
    .clk_i, .rst_ni, .valid_i(e_valid_q),
    .a_i({e_ws_q, 16'h0000}), .b_i({e_xs_q, 16'h0000}),
    .valid_o(), .y_o(e_sc)
  );

  bpu_delay #(.Width(32), .Depth(ScLat - I2fLat)) u_align_isum (
    .clk_i, .rst_ni, .d_i(e_isum_f), .q_o(f_isum)
  );
  bpu_delay #(.Width(32), .Depth(ScLat - MulLat)) u_align_sc (
    .clk_i, .rst_ni, .d_i(e_sc), .q_o(f_sc)
  );
  bpu_delay #(.Width(1), .Depth(ScLat), .Reset(1'b1)) u_fv (
    .clk_i, .rst_ni, .d_i(e_valid_q), .q_o(f_valid)
  );
  bpu_delay #(.Width(2 + RW), .Depth(ScLat)) u_ftag (
    .clk_i, .rst_ni,
    .d_i({e_first_g_q, e_last_g_q, e_r_q}),
    .q_o({f_first_g,   f_last_g,   f_r})
  );

  // ---------------------------------------------------------------------------
  // Stage P: p = isum * scale
  // ---------------------------------------------------------------------------
  logic [31:0]   p;
  logic          p_valid, p_first_g, p_last_g;
  logic [RW-1:0] p_r;

  bpu_fp32_mul #(.PipeMask(MulPipe)) u_prod_mul (
    .clk_i, .rst_ni, .valid_i(f_valid), .a_i(f_isum), .b_i(f_sc),
    .valid_o(p_valid), .y_o(p)
  );
  bpu_delay #(.Width(2 + RW), .Depth(MulLat)) u_ptag (
    .clk_i, .rst_ni,
    .d_i({f_first_g, f_last_g, f_r}),
    .q_o({p_first_g, p_last_g, p_r})
  );

  // ---------------------------------------------------------------------------
  // Stage S: fp32 accumulate, strictly in group order per row
  // ---------------------------------------------------------------------------
  logic [R-1:0][31:0] acc_q;
  logic [31:0]   acc_in, s_y;
  logic          s_valid, s_last_g;
  logic [RW-1:0] s_r;
  logic          ofifo_ready;

  // The first group adds to +0.0 (not a bypass) so signed zeros match the spec.
  assign acc_in = p_first_g ? 32'h0000_0000 : acc_q[p_r];

  bpu_fp32_add #(.PipeMask(AddPipe)) u_acc_add (
    .clk_i, .rst_ni, .valid_i(p_valid), .a_i(acc_in), .b_i(p),
    .valid_o(s_valid), .y_o(s_y)
  );
  bpu_delay #(.Width(1 + RW), .Depth(AddLat)) u_stag (
    .clk_i, .rst_ni, .d_i({p_last_g, p_r}), .q_o({s_last_g, s_r})
  );

  always_ff @(posedge clk_i) begin
    if (s_valid) acc_q[s_r] <= s_y;
  end

  bpu_fifo #(.Width(32), .Depth(OutFifoDepth)) u_ofifo (
    .clk_i, .rst_ni,
    .in_valid_i(s_valid && s_last_g), .in_ready_o(ofifo_ready), .in_data_i(s_y),
    .out_valid_o(y_valid_o), .out_ready_i(y_ready_i), .out_data_o(y_data_o),
    .count_o()
  );

  // ---------------------------------------------------------------------------
  // Simulation-only checks
  // ---------------------------------------------------------------------------
`ifndef SYNTHESIS
  /* verilator lint_off SYNCASYNCNET */
  initial begin
    if (Lanes < 2 || Lanes > G || (Lanes & (Lanes - 1)) != 0)
      $fatal(1, "bpu_qmv_slice: Lanes=%0d must be a power of two in [2, %0d]", Lanes, G);
    if (MaxK % G != 0 || MaxK < 2 * G)
      $fatal(1, "bpu_qmv_slice: MaxK=%0d must be a multiple of %0d and >= %0d", MaxK, G, 2 * G);
    if (R < 1 || OutFifoDepth < R)
      $fatal(1, "bpu_qmv_slice: need RowInterleave >= 1 and OutFifoDepth >= RowInterleave");
    // A row's accumulator must be written back before its next group arrives.
    if (R * Cpg4 < AddLat + 1)
      $fatal(1, "bpu_qmv_slice: RowInterleave*(64/Lanes)=%0d must be >= AddLatency+1=%0d",
             R * Cpg4, AddLat + 1);
  end

  always @(posedge clk_i) begin
    if (rst_ni) begin
      if (s_valid && s_last_g && !ofifo_ready)
        $error("bpu_qmv_slice: output FIFO overflow (credit accounting broken)");
      if (cmd_fire && (cmd_ngroups_i == '0 || cmd_ngroups_i > GW'(MaxK / G) || cmd_nrowblk_i == '0))
        $error("bpu_qmv_slice: illegal command ngroups=%0d nrowblk=%0d", cmd_ngroups_i, cmd_nrowblk_i);
      if (run_q && (x_we_i || xs_we_i))
        $error("bpu_qmv_slice: activation buffer written while an operation is streaming");
    end
  end
  /* verilator lint_on SYNCASYNCNET */
`endif

endmodule
