// Compute top: the QMV array and the FVU sharing the FVU scratchpad, driven by a
// stream of operations (the compute side of the BPU's command interface).
//
// An operation is either
//   FVU (cmd_qmv_i = 0): one bpuref.fvu op, passed to bpu_fvu, or
//   QMV (cmd_qmv_i = 1): y[0:n] = W x (bpuref.qwen.QmvOp):
//     1. load x: K int8 codes (stored in the SPM as exact fp32 integers) and
//        K/64 bf16 scales (stored as fp32) into the array's activation buffers,
//     2. start the array and ask the memory side for weight tensor q_wid
//        (wreq_*); the memory side then streams layout-L0 beats per slice,
//     3. write the results back into the SPM: y[i] for i < n, or, in argmax
//        mode, y[0] = index (as fp32) and y[1] = value.
// Operations run strictly in order, one at a time. The host SPM port works
// while cmd_ready_o is high.
module bpu_compute_top #(
  // QMV array
  parameter int unsigned NSlice        = 1,
  parameter int unsigned Lanes         = 16,
  parameter int unsigned RowInterleave = 1,
  parameter int unsigned MaxK          = 2048,
  parameter int unsigned RowBlkW       = 12,
  parameter int unsigned IdxW          = 20,
  parameter bit          QProdReg      = 1'b1,
  parameter int unsigned QTreeRegEvery = 0,
  parameter bit          QI2fReg       = 1'b1,
  parameter logic [2:0]  QMulPipe      = 3'b010,
  parameter logic [2:0]  QAddPipe      = 3'b010,
  // FVU
  parameter int unsigned VLanes        = 2,
  parameter int unsigned SpmWords      = 32768,
  parameter logic [2:0]  FMulPipe      = 3'b010,
  parameter logic [2:0]  FAddPipe      = 3'b010,
  parameter logic [4:0]  FSfuPipe      = 5'b01010,
  parameter int unsigned RedFifoDepth  = 8
) (
  input  logic                                clk_i,
  input  logic                                rst_ni,

  // Operation stream
  input  logic                                cmd_valid_i,
  output logic                                cmd_ready_o,
  input  logic                                cmd_qmv_i,
  //   FVU fields
  input  logic [4:0]                          cmd_op_i,
  input  logic [2:0]                          cmd_func_i,
  input  logic [4:0]                          cmd_half_log2_i,
  input  logic [15:0]                         cmd_rows_i,
  input  logic [15:0]                         cmd_cols_i,
  input  logic [$clog2(SpmWords*VLanes)-1:0]  cmd_d_i, cmd_a_i, cmd_b_i, cmd_c_i, cmd_s_i, cmd_t_i,
  input  logic [$clog2(SpmWords*VLanes)-1:0]  cmd_ds_i, cmd_as_i, cmd_bs_i, cmd_cs_i, cmd_ss_i, cmd_ts_i,
  //   QMV fields
  input  logic [15:0]                         cmd_wid_i,
  input  logic                                cmd_wfmt_i,
  input  logic                                cmd_argmax_i,
  input  logic [15:0]                         cmd_k_i,
  input  logic [IdxW-1:0]                     cmd_n_i,
  input  logic [$clog2(SpmWords*VLanes)-1:0]  cmd_x_i, cmd_xs_i, cmd_y_i,

  // Weight request to the memory side, then per-slice weight/scale streams
  output logic                                wreq_valid_o,
  input  logic                                wreq_ready_i,
  output logic [15:0]                         wreq_id_o,
  output logic [RowBlkW-1:0]                  wreq_nrowblk_o,
  output logic [$clog2(MaxK/64+1)-1:0]        wreq_ngroups_o,
  output logic                                wreq_wfmt_o,
  input  logic [NSlice-1:0]                   w_valid_i,
  output logic [NSlice-1:0]                   w_ready_o,
  input  logic [NSlice*Lanes*4-1:0]           w_data_i,
  input  logic [NSlice-1:0]                   ws_valid_i,
  output logic [NSlice-1:0]                   ws_ready_o,
  input  logic [NSlice*16-1:0]                ws_data_i,

  // Host SPM port (only while cmd_ready_o is high)
  input  logic                                host_we_i,
  input  logic [$clog2(SpmWords)-1:0]         host_waddr_i,
  input  logic [VLanes-1:0]                   host_wmask_i,
  input  logic [VLanes*32-1:0]                host_wdata_i,
  input  logic                                host_re_i,
  input  logic [$clog2(SpmWords)-1:0]         host_raddr_i,
  output logic [VLanes*32-1:0]                host_rdata_o,

  output logic                                busy_o,
  input  logic                                status_clr_i,
  output logic                                err_o,
  output logic [31:0]                         perf_ops_o
);

  import bpu_compute_pkg::*;

  localparam int unsigned V     = VLanes;
  localparam int unsigned LW    = $clog2(V);
  localparam int unsigned AW    = $clog2(SpmWords * V);
  localparam int unsigned WAW   = $clog2(SpmWords);
  localparam int unsigned GW    = $clog2(MaxK / 64 + 1);
  localparam int unsigned Chunk = (V < Lanes) ? V : Lanes;     // gearbox step (elements)
  localparam int unsigned RowsPerBlk = NSlice * RowInterleave;

  // Operation registers (captured when a QMV operation is accepted)
  logic                    wfmt_q, argmax_q;
  logic [15:0]             k_q, wid_q;
  logic [IdxW-1:0]         n_q, got_q;
  logic [AW-1:0]           x_q, xs_q, y_q;

  // ---------------------------------------------------------------------------
  // Engines
  // ---------------------------------------------------------------------------
  logic                    f_cmd_valid, f_cmd_ready, f_busy, f_err;
  logic                    f_ext_we, f_ext_re;
  logic [WAW-1:0]          f_ext_waddr, f_ext_raddr;
  logic [V-1:0]            f_ext_wmask;
  logic [V*32-1:0]         f_ext_wdata, f_ext_rdata;

  bpu_fvu #(
    .VLanes(V), .SpmWords(SpmWords), .MulPipe(FMulPipe), .AddPipe(FAddPipe),
    .SfuPipe(FSfuPipe), .RedFifoDepth(RedFifoDepth), .EnPerf(1'b0)
  ) u_fvu (
    .clk_i, .rst_ni,
    .cmd_valid_i(f_cmd_valid), .cmd_ready_o(f_cmd_ready),
    .cmd_op_i, .cmd_func_i, .cmd_half_log2_i, .cmd_rows_i, .cmd_cols_i,
    .cmd_d_i, .cmd_a_i, .cmd_b_i, .cmd_c_i, .cmd_s_i, .cmd_t_i,
    .cmd_ds_i, .cmd_as_i, .cmd_bs_i, .cmd_cs_i, .cmd_ss_i, .cmd_ts_i,
    .ext_we_i(f_ext_we), .ext_waddr_i(f_ext_waddr), .ext_wmask_i(f_ext_wmask),
    .ext_wdata_i(f_ext_wdata), .ext_re_i(f_ext_re), .ext_raddr_i(f_ext_raddr),
    .ext_rdata_o(f_ext_rdata),
    .busy_o(f_busy), .status_clr_i, .err_cmd_o(f_err), .perf_items_o(), .perf_cycles_o()
  );

  logic                    q_x_we, q_xs_we, q_cmd_valid, q_cmd_ready, q_y_valid, q_y_ready;
  logic                    q_busy, q_err;
  logic [$clog2(MaxK/Lanes)-1:0] q_x_waddr;
  logic [Lanes*8-1:0]      q_x_wdata;
  logic [$clog2(MaxK/64)-1:0] q_xs_waddr;
  logic [15:0]             q_xs_wdata;
  logic [31:0]             q_y_data;
  logic [IdxW-1:0]         q_y_index;
  logic [RowBlkW-1:0]      q_nrowblk;

  bpu_qmv_array #(
    .NSlice(NSlice), .Lanes(Lanes), .RowInterleave(RowInterleave), .MaxK(MaxK),
    .RowBlkW(RowBlkW), .IdxW(IdxW), .ProdReg(QProdReg), .TreeRegEvery(QTreeRegEvery),
    .I2fReg(QI2fReg), .MulPipe(QMulPipe), .AddPipe(QAddPipe), .EnPerf(1'b0)
  ) u_qmv (
    .clk_i, .rst_ni,
    .x_we_i(q_x_we), .x_waddr_i(q_x_waddr), .x_wdata_i(q_x_wdata),
    .xs_we_i(q_xs_we), .xs_waddr_i(q_xs_waddr), .xs_wdata_i(q_xs_wdata),
    .cmd_valid_i(q_cmd_valid), .cmd_ready_o(q_cmd_ready), .cmd_wfmt_i(wfmt_q),
    .cmd_ngroups_i(GW'(k_q >> 6)), .cmd_nrowblk_i(q_nrowblk), .cmd_argmax_i(argmax_q),
    .cmd_nrows_i(n_q),
    .w_valid_i, .w_ready_o, .w_data_i, .ws_valid_i, .ws_ready_o, .ws_data_i,
    .y_valid_o(q_y_valid), .y_ready_i(q_y_ready), .y_data_o(q_y_data), .y_index_o(q_y_index),
    .busy_o(q_busy), .status_clr_i, .err_cmd_o(q_err), .flag_nan_o(), .flag_inf_o(),
    .perf_beats_o(), .perf_stall_w_o(), .perf_stall_ws_o(), .perf_stall_out_o()
  );

  // ---------------------------------------------------------------------------
  // Dispatcher
  // ---------------------------------------------------------------------------
  typedef enum logic [2:0] {DIdle, DFvu, DLoadX, DLoadXs, DIssue, DCollect, DArgIdx} dstate_e;
  dstate_e st_q;

  logic [15:0]             ck_q;                 // gearbox chunk / scale index
  logic                    rd_v_q;               // an SPM read is returning this cycle
  logic [15:0]             rd_ck_q;
  logic                    q_cmd_done_q, wreq_done_q;
  logic [31:0]             arg_val_q;
  logic [Lanes*8-1:0]      xword_q;

  logic                    fire, idle, qmv_legal, err_q;
  assign idle        = (st_q == DIdle);
  // The dispatcher checks QMV operations itself: an operation the array would reject
  // must not reach the memory side, which would then stream weights nobody takes.
  assign qmv_legal   = (cmd_k_i != '0) && (cmd_k_i[5:0] == '0) && (32'(cmd_k_i) <= MaxK)
                    && (cmd_n_i != '0) && (cmd_x_i[5:0] == '0)
                    && ((32'(cmd_n_i) + RowsPerBlk - 1) / RowsPerBlk < (32'd1 << RowBlkW));
  assign cmd_ready_o = idle && f_cmd_ready && q_cmd_ready;
  assign fire        = cmd_valid_i && cmd_ready_o;
  assign f_cmd_valid = fire && !cmd_qmv_i;
  assign busy_o      = !idle || f_busy || q_busy;
  assign err_o       = f_err || q_err || err_q;
  assign q_nrowblk   = RowBlkW'((32'(n_q) + RowsPerBlk - 1) / RowsPerBlk);

  // Gearbox: one chunk of min(VLanes, Lanes) codes per cycle, SPM -> QMV x buffer.
  logic [15:0] nchunks;
  logic [AW-1:0] ck_el, rd_el;
  assign nchunks = 16'(k_q / Chunk);
  assign ck_el   = x_q + AW'(ck_q) * AW'(Chunk);
  logic ck_needs_read;
  assign ck_needs_read = (st_q == DLoadX) && (ck_q < nchunks);

  // Scales: one per cycle.
  logic sc_needs_read;
  assign sc_needs_read = (st_q == DLoadXs) && (ck_q < 16'(k_q >> 6));
  assign rd_el = (st_q == DLoadX) ? ck_el : xs_q + AW'(ck_q);

  // Returned data (1-cycle SPM read): lane of the element that was requested.
  // x is 64-aligned, so a chunk starts on a chunk-aligned lane.
  logic [LW-1:0] ret_lane_q;
  always_ff @(posedge clk_i) ret_lane_q <= rd_el[LW-1:0];

  // Chunk of codes out of the returned SPM word (fp32 integers -> int8)
  logic [Chunk*8-1:0] ck_bytes;
  for (genvar c = 0; c < Chunk; c++) begin : g_ck
    logic [31:0] e;
    assign e = f_ext_rdata[(int'(ret_lane_q) + c)*32 +: 32];
    assign ck_bytes[c*8 +: 8] = f32_to_i8(e);
  end

  // QMV activation buffer writes
  localparam int unsigned PerWord = Lanes / Chunk;      // chunks per QMV word
  logic [$clog2(PerWord+1)-1:0] in_word;
  assign in_word = ($bits(in_word))'(rd_ck_q % PerWord);

  always_comb begin
    q_x_we     = 1'b0;
    q_x_waddr  = ($bits(q_x_waddr))'(rd_ck_q / PerWord);
    q_x_wdata  = xword_q;
    q_x_wdata[int'(in_word)*Chunk*8 +: Chunk*8] = ck_bytes;
    if (st_q == DLoadX && rd_v_q && (in_word == ($bits(in_word))'(PerWord - 1))) q_x_we = 1'b1;
  end

  always_ff @(posedge clk_i) begin
    if (st_q == DLoadX && rd_v_q) xword_q <= q_x_wdata;
  end

  assign q_xs_we    = (st_q == DLoadXs) && rd_v_q;
  assign q_xs_waddr = ($bits(q_xs_waddr))'(rd_ck_q);
  assign q_xs_wdata = f_ext_rdata[int'(ret_lane_q)*32 + 16 +: 16];    // bf16 = top half

  // SPM port: host while idle, the dispatcher during QMV loads and write-back.
  logic          d_we;
  logic [AW-1:0] d_wel;
  logic [31:0]   d_wdata;

  always_comb begin
    f_ext_re    = idle ? host_re_i : (ck_needs_read || sc_needs_read);
    f_ext_raddr = idle ? host_raddr_i : WAW'(rd_el >> LW);
    f_ext_we    = idle ? host_we_i : d_we;
    f_ext_waddr = idle ? host_waddr_i : WAW'(d_wel >> LW);
    f_ext_wmask = idle ? host_wmask_i : (V'(1) << d_wel[LW-1:0]);
    f_ext_wdata = idle ? host_wdata_i : {V{d_wdata}};
  end
  assign host_rdata_o = f_ext_rdata;

  // Results -> SPM (one element per cycle)
  logic [31:0] idx_f;
  bpu_int2fp32 #(.InW(IdxW + 1), .Reg(1'b0)) u_idx2f (
    .clk_i, .rst_ni, .valid_i(1'b1), .x_i({1'b0, q_y_index}), .valid_o(), .y_o(idx_f));

  assign q_y_ready = (st_q == DCollect);
  always_comb begin
    d_we    = 1'b0;
    d_wel   = y_q + AW'(q_y_index);
    d_wdata = q_y_data;
    if (st_q == DCollect && q_y_valid) begin
      d_we = 1'b1;
      if (argmax_q) begin
        d_wel   = y_q;
        d_wdata = idx_f;
      end
    end else if (st_q == DArgIdx) begin
      d_we    = 1'b1;
      d_wel   = y_q + AW'(1);
      d_wdata = arg_val_q;
    end
  end

  assign q_cmd_valid  = (st_q == DIssue) && !q_cmd_done_q;
  assign wreq_valid_o = (st_q == DIssue) && !wreq_done_q;
  assign wreq_id_o      = wid_q;
  assign wreq_nrowblk_o = q_nrowblk;
  assign wreq_ngroups_o = GW'(k_q >> 6);
  assign wreq_wfmt_o    = wfmt_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      st_q         <= DIdle;
      ck_q         <= '0;
      rd_v_q       <= 1'b0;
      rd_ck_q      <= '0;
      q_cmd_done_q <= 1'b0;
      wreq_done_q  <= 1'b0;
      got_q        <= '0;
      perf_ops_o   <= '0;
      err_q        <= 1'b0;
    end else begin
      if (status_clr_i)                          err_q <= 1'b0;
      else if (fire && cmd_qmv_i && !qmv_legal)  err_q <= 1'b1;
      rd_v_q  <= ck_needs_read || sc_needs_read;
      rd_ck_q <= ck_q;
      if (status_clr_i) perf_ops_o <= '0;
      else if (fire)    perf_ops_o <= perf_ops_o + 1'b1;
      unique case (st_q)
        DIdle: begin
          if (fire && (!cmd_qmv_i || qmv_legal)) begin
            st_q <= cmd_qmv_i ? DLoadX : DFvu;
            ck_q <= '0;
          end
        end
        DFvu: begin                    // FVU accepted the op at `fire`; wait for it to finish
          if (f_cmd_ready) st_q <= DIdle;
        end
        DLoadX: begin
          if (ck_q < nchunks) ck_q <= ck_q + 1'b1;
          else if (!rd_v_q) begin
            st_q <= DLoadXs;
            ck_q <= '0;
          end
        end
        DLoadXs: begin
          if (ck_q < 16'(k_q >> 6)) ck_q <= ck_q + 1'b1;
          else if (!rd_v_q) begin
            st_q         <= DIssue;
            q_cmd_done_q <= 1'b0;
            wreq_done_q  <= 1'b0;
          end
        end
        DIssue: begin
          if (q_cmd_valid && q_cmd_ready) q_cmd_done_q <= 1'b1;
          if (wreq_valid_o && wreq_ready_i) wreq_done_q <= 1'b1;
          if ((q_cmd_done_q || q_cmd_ready) && (wreq_done_q || wreq_ready_i)) begin
            st_q  <= DCollect;
            got_q <= '0;
          end
        end
        DCollect: begin
          if (q_y_valid) begin
            got_q <= got_q + 1'b1;
            if (argmax_q) begin
              st_q      <= DArgIdx;
              arg_val_q <= q_y_data;
            end else if (got_q == n_q - 1'b1) begin
              st_q <= DIdle;
            end
          end
        end
        DArgIdx: st_q <= DIdle;
        default: st_q <= DIdle;
      endcase
    end
  end

  always_ff @(posedge clk_i) begin
    if (fire && cmd_qmv_i) begin
      wid_q    <= cmd_wid_i;
      wfmt_q   <= cmd_wfmt_i;
      argmax_q <= cmd_argmax_i;
      k_q      <= cmd_k_i;
      n_q      <= cmd_n_i;
      x_q      <= cmd_x_i;
      xs_q     <= cmd_xs_i;
      y_q      <= cmd_y_i;
    end
  end

`ifndef SYNTHESIS
  initial begin
    if (Lanes % Chunk != 0 || V % Chunk != 0)
      $fatal(1, "bpu_compute_top: VLanes and Lanes must be powers of two");
  end
`endif

endmodule
