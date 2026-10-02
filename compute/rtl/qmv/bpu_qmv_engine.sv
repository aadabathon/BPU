// QMV engine: the QMV array as a shared-SRAM client (the "matrix unit").
//
// One command y[0:n] = W x (bpuref.qwen.QmvOp):
//   load     read K int8 codes (stored as exact fp32 integers at x, 64-aligned) and
//            K/64 bf16 scales (stored as fp32 at xs) from the shared SRAM into the
//            array's activation buffers. Reads are request/response with credits
//            for a LdDepth-word buffer; a gearbox moves min(VLanes, Lanes) codes and
//            one scale per cycle. The activation buffers stay inside the engine:
//            every row of W re-reads x, which the shared SRAM could not feed.
//   issue    start the array and ask the memory side for weight tensor wid
//            (wreq_*); the memory side then streams layout-L0 beats per slice.
//   collect  pack results into SRAM words and write them: y[i] for i < n, or in
//            argmax mode y[0] = index (as fp32) and y[1] = value.
// done_o pulses when the last result write has been accepted; err_o qualifies it
// (malformed command, consumed without running; or an out-of-range address).
module bpu_qmv_engine #(
  // QMV array
  parameter int unsigned NSlice        = 1,
  parameter int unsigned Lanes         = 16,
  parameter int unsigned RowInterleave = 1,
  parameter int unsigned MaxK          = 2048,
  parameter int unsigned RowBlkW       = 12,
  parameter int unsigned IdxW          = 20,
  parameter bit          ProdReg       = 1'b1,
  parameter int unsigned TreeRegEvery  = 0,
  parameter bit          I2fReg        = 1'b1,
  parameter logic [2:0]  MulPipe       = 3'b010,
  parameter logic [2:0]  AddPipe       = 3'b010,
  // Shared SRAM words
  parameter int unsigned VLanes        = 4,
  parameter int unsigned LdDepth       = 4
) (
  input  logic                          clk_i,
  input  logic                          rst_ni,

  input  logic                          cmd_valid_i,
  output logic                          cmd_ready_o,
  input  logic [15:0]                   cmd_wid_i,
  input  logic                          cmd_wfmt_i,
  input  logic                          cmd_argmax_i,
  input  logic [15:0]                   cmd_k_i,
  input  logic [IdxW-1:0]               cmd_n_i,
  input  logic [31:0]                   cmd_x_i, cmd_xs_i, cmd_y_i,
  output logic                          done_o,
  output logic                          err_o,

  // Shared SRAM
  output logic                          mrd_valid_o,
  input  logic                          mrd_ready_i,
  output logic [31-$clog2(VLanes):0]    mrd_addr_o,
  input  logic                          mrd_rvalid_i,
  input  logic [VLanes*32-1:0]          mrd_rdata_i,
  input  logic                          mrd_oob_i,
  output logic                          mwr_valid_o,
  input  logic                          mwr_ready_i,
  output logic [31-$clog2(VLanes):0]    mwr_addr_o,
  output logic [VLanes-1:0]             mwr_mask_o,
  output logic [VLanes*32-1:0]          mwr_data_o,
  input  logic                          mwr_oob_i,

  // Weight request to the memory side, then per-slice weight/scale streams
  output logic                          wreq_valid_o,
  input  logic                          wreq_ready_i,
  output logic [15:0]                   wreq_id_o,
  output logic [RowBlkW-1:0]            wreq_nrowblk_o,
  output logic [$clog2(MaxK/64+1)-1:0]  wreq_ngroups_o,
  output logic                          wreq_wfmt_o,
  input  logic [NSlice-1:0]             w_valid_i,
  output logic [NSlice-1:0]             w_ready_o,
  input  logic [NSlice*Lanes*4-1:0]     w_data_i,
  input  logic [NSlice-1:0]             ws_valid_i,
  output logic [NSlice-1:0]             ws_ready_o,
  input  logic [NSlice*16-1:0]          ws_data_i,

  output logic                          busy_o,
  output logic                          flag_nan_o,     // since the last command started
  output logic                          flag_inf_o
);

  import bpu_compute_pkg::*;

  localparam int unsigned V     = VLanes;
  localparam int unsigned LW    = $clog2(V);
  localparam int unsigned MAW   = 32 - LW;
  localparam int unsigned GW    = $clog2(MaxK / 64 + 1);
  localparam int unsigned Chunk = (V < Lanes) ? V : Lanes;     // codes per gearbox step
  localparam int unsigned SubN  = V / Chunk;                   // gearbox steps per SRAM word
  localparam int unsigned SubW  = (SubN > 1) ? $clog2(SubN) : 1;
  localparam int unsigned PerX  = Lanes / Chunk;               // gearbox steps per x-buffer word
  localparam int unsigned RowsPerBlk = NSlice * RowInterleave;
  localparam int unsigned XAW   = $clog2(MaxK / Lanes);
  localparam int unsigned SAW   = $clog2(MaxK / 64);

  typedef enum logic [2:0] {EIdle, ELoad, EIssue, ECollect, EArgVal, EFlush} estate_e;
  estate_e st_q;

  // Command registers
  logic             wfmt_q, argmax_q, err_q;
  logic [15:0]      k_q, wid_q;
  logic [IdxW-1:0]  n_q;
  logic [31:0]      x_q, xs_q, y_q;

  logic cmd_fire, qmv_legal;
  logic q_cmd_ready;                                  // the array takes a new command
  // An operation the array would reject must not reach the memory side, which would
  // then stream weights nobody takes.
  assign qmv_legal = (cmd_k_i != '0) && (cmd_k_i[5:0] == '0) && (32'(cmd_k_i) <= MaxK)
                  && (cmd_n_i != '0) && (cmd_x_i[5:0] == '0)
                  && ((32'(cmd_n_i) + RowsPerBlk - 1) / RowsPerBlk < (32'd1 << RowBlkW));
  // The array may still be computing the padding rows of the last row block after
  // the real results are out: a new command waits until it is ready again.
  assign cmd_ready_o = (st_q == EIdle) && q_cmd_ready;
  assign cmd_fire    = cmd_valid_i && cmd_ready_o;

  logic [15:0] ngroups, nxw, nsw;
  logic [31:0] xs_last;
  assign ngroups = k_q >> 6;
  assign nxw     = 16'(k_q >> LW);                                  // x words
  assign xs_last = xs_q + 32'(ngroups) - 1;
  assign nsw     = 16'((xs_last >> LW) - (xs_q >> LW) + 1);          // scale words

  // ---------------------------------------------------------------------------
  // Array
  // ---------------------------------------------------------------------------
  logic                    q_x_we, q_xs_we, q_cmd_valid, q_y_valid, q_y_ready;
  logic                    q_busy, q_err;
  logic [XAW-1:0]          q_x_waddr;
  logic [Lanes*8-1:0]      q_x_wdata;
  logic [SAW-1:0]          q_xs_waddr;
  logic [15:0]             q_xs_wdata;
  logic [31:0]             q_y_data;
  logic [IdxW-1:0]         q_y_index;
  logic [RowBlkW-1:0]      q_nrowblk;
  assign q_nrowblk = RowBlkW'((32'(n_q) + RowsPerBlk - 1) / RowsPerBlk);

  bpu_qmv_array #(
    .NSlice(NSlice), .Lanes(Lanes), .RowInterleave(RowInterleave), .MaxK(MaxK),
    .RowBlkW(RowBlkW), .IdxW(IdxW), .ProdReg(ProdReg), .TreeRegEvery(TreeRegEvery),
    .I2fReg(I2fReg), .MulPipe(MulPipe), .AddPipe(AddPipe), .EnPerf(1'b0)
  ) u_array (
    .clk_i, .rst_ni,
    .x_we_i(q_x_we), .x_waddr_i(q_x_waddr), .x_wdata_i(q_x_wdata),
    .xs_we_i(q_xs_we), .xs_waddr_i(q_xs_waddr), .xs_wdata_i(q_xs_wdata),
    .cmd_valid_i(q_cmd_valid), .cmd_ready_o(q_cmd_ready), .cmd_wfmt_i(wfmt_q),
    .cmd_ngroups_i(GW'(ngroups)), .cmd_nrowblk_i(q_nrowblk), .cmd_argmax_i(argmax_q),
    .cmd_nrows_i(n_q),
    .w_valid_i, .w_ready_o, .w_data_i, .ws_valid_i, .ws_ready_o, .ws_data_i,
    .y_valid_o(q_y_valid), .y_ready_i(q_y_ready), .y_data_o(q_y_data), .y_index_o(q_y_index),
    .busy_o(q_busy), .status_clr_i(cmd_fire), .err_cmd_o(q_err), .flag_nan_o, .flag_inf_o,
    .perf_beats_o(), .perf_stall_w_o(), .perf_stall_ws_o(), .perf_stall_out_o());

  // ---------------------------------------------------------------------------
  // Load: request words (x words, then scale words), credits for the buffer
  // ---------------------------------------------------------------------------
  logic [15:0] rq_q;                                  // words requested
  logic [$clog2(LdDepth+1)-1:0] ld_cred_q;
  logic        ld_pop, ld_have, ld_ready;
  logic [V*32-1:0] ld_word;

  assign mrd_valid_o = (st_q == ELoad) && (rq_q < nxw + nsw) && (ld_cred_q != '0);
  assign mrd_addr_o  = (rq_q < nxw) ? MAW'((x_q >> LW) + 32'(rq_q))
                                    : MAW'((xs_q >> LW) + 32'(rq_q - nxw));

  bpu_fifo #(.Width(V*32), .Depth(LdDepth)) u_ld (
    .clk_i, .rst_ni,
    .in_valid_i(mrd_rvalid_i), .in_ready_o(ld_ready), .in_data_i(mrd_rdata_i),
    .out_valid_o(ld_have), .out_ready_i(ld_pop), .out_data_o(ld_word), .count_o());

  // Gearbox: x words first (Chunk codes per step), then scales (one per step)
  logic [15:0]       cw_q;                            // SRAM words consumed
  logic [SubW-1:0]   sub_q;                           // step within an x word
  logic [15:0]       xstep_q;                         // x gearbox steps done
  logic [15:0]       sj_q;                            // scales written
  logic [Lanes*8-1:0] xacc_q;
  logic              in_x, x_done, s_done, step;
  logic [LW-1:0]     s_lane;
  logic [Chunk*8-1:0] ck_bytes;

  assign in_x   = (cw_q < nxw);
  assign x_done = !in_x;
  assign s_done = (sj_q == ngroups);
  assign step   = (st_q == ELoad) && ld_have && !(x_done && s_done);
  assign s_lane = LW'(xs_q + 32'(sj_q));

  for (genvar c = 0; c < Chunk; c++) begin : g_ck
    assign ck_bytes[c*8 +: 8] = f32_to_i8(ld_word[(int'(sub_q) * Chunk + c)*32 +: 32]);
  end

  logic [$clog2(PerX+1)-1:0] in_xw;
  assign in_xw = ($bits(in_xw))'(xstep_q % PerX);
  always_comb begin
    q_x_wdata = xacc_q;
    q_x_wdata[int'(in_xw)*Chunk*8 +: Chunk*8] = ck_bytes;
  end
  assign q_x_we     = step && in_x && (in_xw == ($bits(in_xw))'(PerX - 1));
  assign q_x_waddr  = XAW'(xstep_q / PerX);
  assign q_xs_we    = step && !in_x;
  assign q_xs_waddr = SAW'(sj_q);
  assign q_xs_wdata = ld_word[int'(s_lane)*32 + 16 +: 16];          // bf16 = top half
  assign ld_pop     = step && (in_x ? (int'(sub_q) == SubN - 1)
                                    : (s_lane == LW'(V - 1) || sj_q == ngroups - 1));

  // ---------------------------------------------------------------------------
  // Collect: pack results into words, one write register in front of the port
  // ---------------------------------------------------------------------------
  logic [IdxW-1:0]   got_q;
  logic              acc_v_q, wr_v_q;
  logic [MAW-1:0]    acc_w_q, wr_w_q;
  logic [V-1:0]      acc_m_q, wr_m_q;
  logic [V*32-1:0]   acc_d_q, wr_d_q;
  logic [31:0]       arg_val_q, idx_f;
  logic              put;                                // place one element this cycle
  logic [31:0]       put_el, put_val;
  logic              wr_free, last_put;

  bpu_int2fp32 #(.InW(IdxW + 1), .Reg(1'b0)) u_idx2f (
    .clk_i, .rst_ni, .valid_i(1'b1), .x_i({1'b0, q_y_index}), .valid_o(), .y_o(idx_f));

  assign wr_free   = !wr_v_q || mwr_ready_i;
  assign q_y_ready = (st_q == ECollect) && wr_free;
  always_comb begin
    put     = 1'b0;
    put_el  = y_q + 32'(q_y_index);
    put_val = q_y_data;
    if (st_q == ECollect && q_y_valid && wr_free) begin
      put = 1'b1;
      if (argmax_q) begin
        put_el  = y_q;
        put_val = idx_f;
      end
    end else if (st_q == EArgVal && wr_free) begin
      put     = 1'b1;
      put_el  = y_q + 32'd1;
      put_val = arg_val_q;
    end
  end
  assign last_put = put && (argmax_q ? (st_q == EArgVal) : (got_q == n_q - 1'b1));

  assign mwr_valid_o = wr_v_q;
  assign mwr_addr_o  = wr_w_q;
  assign mwr_mask_o  = wr_m_q;
  assign mwr_data_o  = wr_d_q;

  // ---------------------------------------------------------------------------
  // Control
  // ---------------------------------------------------------------------------
  logic q_cmd_done_q, wreq_done_q;
  assign q_cmd_valid  = (st_q == EIssue) && !q_cmd_done_q;
  assign wreq_valid_o = (st_q == EIssue) && !wreq_done_q;
  assign wreq_id_o      = wid_q;
  assign wreq_nrowblk_o = q_nrowblk;
  assign wreq_ngroups_o = GW'(ngroups);
  assign wreq_wfmt_o    = wfmt_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      st_q         <= EIdle;
      rq_q         <= '0;
      ld_cred_q    <= ($bits(ld_cred_q))'(LdDepth);
      cw_q         <= '0;
      sub_q        <= '0;
      xstep_q      <= '0;
      sj_q         <= '0;
      q_cmd_done_q <= 1'b0;
      wreq_done_q  <= 1'b0;
      got_q        <= '0;
      acc_v_q      <= 1'b0;
      wr_v_q       <= 1'b0;
      err_q        <= 1'b0;
      done_o       <= 1'b0;
      err_o        <= 1'b0;
    end else begin
      done_o <= 1'b0;
      err_o  <= 1'b0;
      ld_cred_q <= ld_cred_q - ($bits(ld_cred_q))'(mrd_valid_o && mrd_ready_i)
                             + ($bits(ld_cred_q))'(ld_pop);
      if (mrd_valid_o && mrd_ready_i) rq_q <= rq_q + 1'b1;
      if ((mrd_oob_i && mrd_valid_o) || (mwr_oob_i && mwr_valid_o) || q_err) err_q <= 1'b1;
      if (step) begin
        if (in_x) begin
          if (int'(sub_q) == SubN - 1) begin
            sub_q <= '0;
            cw_q  <= cw_q + 1'b1;
          end else begin
            sub_q <= sub_q + 1'b1;
          end
          xstep_q <= xstep_q + 1'b1;
        end else begin
          sj_q <= sj_q + 1'b1;
          if (ld_pop) cw_q <= cw_q + 1'b1;
        end
      end
      // Results: a new word flushes the previous one into the write register.
      if (wr_v_q && mwr_ready_i) wr_v_q <= 1'b0;
      if (put) begin
        if (acc_v_q && acc_w_q != MAW'(put_el >> LW)) begin
          wr_v_q  <= 1'b1;
          wr_w_q  <= acc_w_q;
          wr_m_q  <= acc_m_q;
          wr_d_q  <= acc_d_q;
          acc_m_q <= V'(1) << put_el[LW-1:0];
        end else begin
          acc_m_q <= (acc_v_q ? acc_m_q : '0) | (V'(1) << put_el[LW-1:0]);
        end
        acc_v_q <= 1'b1;
        acc_w_q <= MAW'(put_el >> LW);
        acc_d_q[int'(put_el[LW-1:0])*32 +: 32] <= put_val;
      end
      unique case (st_q)
        EIdle: begin
          if (cmd_fire) begin
            err_q <= 1'b0;
            if (qmv_legal) begin
              st_q    <= ELoad;
              rq_q    <= '0;
              cw_q    <= '0;
              sub_q   <= '0;
              xstep_q <= '0;
              sj_q    <= '0;
            end else begin
              done_o <= 1'b1;
              err_o  <= 1'b1;
            end
          end
        end
        ELoad: begin
          if (x_done && s_done) begin
            st_q         <= EIssue;
            q_cmd_done_q <= 1'b0;
            wreq_done_q  <= 1'b0;
          end
        end
        EIssue: begin
          if (q_cmd_valid && q_cmd_ready) q_cmd_done_q <= 1'b1;
          if (wreq_valid_o && wreq_ready_i) wreq_done_q <= 1'b1;
          if ((q_cmd_done_q || q_cmd_ready) && (wreq_done_q || wreq_ready_i)) begin
            st_q    <= ECollect;
            got_q   <= '0;
            acc_v_q <= 1'b0;
          end
        end
        ECollect: begin
          if (put) begin
            got_q <= got_q + 1'b1;
            if (argmax_q) begin
              st_q      <= EArgVal;
              arg_val_q <= q_y_data;
            end else if (last_put) begin
              st_q <= EFlush;
            end
          end
        end
        EArgVal: if (put) st_q <= EFlush;
        EFlush: begin
          // Move the last partial word to the write register, then wait for it.
          if (acc_v_q && wr_free) begin
            wr_v_q  <= 1'b1;
            wr_w_q  <= acc_w_q;
            wr_m_q  <= acc_m_q;
            wr_d_q  <= acc_d_q;
            acc_v_q <= 1'b0;
          end else if (!acc_v_q && (!wr_v_q || mwr_ready_i)) begin
            st_q   <= EIdle;
            done_o <= 1'b1;
            err_o  <= err_q || (mwr_oob_i && mwr_valid_o);
          end
        end
        default: st_q <= EIdle;
      endcase
    end
  end

  // Command capture, and the x accumulator for the gearbox
  always_ff @(posedge clk_i) begin
    if (cmd_fire) begin
      wid_q    <= cmd_wid_i;
      wfmt_q   <= cmd_wfmt_i;
      argmax_q <= cmd_argmax_i;
      k_q      <= cmd_k_i;
      n_q      <= cmd_n_i;
      x_q      <= cmd_x_i;
      xs_q     <= cmd_xs_i;
      y_q      <= cmd_y_i;
    end
    if (step && in_x) xacc_q <= q_x_wdata;
  end

  assign busy_o = (st_q != EIdle) || q_busy;

`ifdef FORMAL
  always_comb begin
    if (rst_ni) begin
      assert (!(mrd_rvalid_i && !ld_ready));            // credits keep the load buffer in bounds
      assert (ld_cred_q <= ($bits(ld_cred_q))'(LdDepth));
    end
  end
`endif

`ifndef SYNTHESIS
  /* verilator lint_off SYNCASYNCNET */
  initial begin
    if (Lanes % Chunk != 0 || V % Chunk != 0)
      $fatal(1, "bpu_qmv_engine: VLanes and Lanes must be powers of two");
  end
  always @(posedge clk_i)
    if (rst_ni && mrd_rvalid_i && !ld_ready) $error("bpu_qmv_engine: load buffer overflow");
  /* verilator lint_on SYNCASYNCNET */
`endif

endmodule
