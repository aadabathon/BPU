// QMV array: NSlice slices computing one matrix-vector product together.
//
// Row striping: global row n is computed by slice n % NSlice (as its local row
// n / NSlice). Every slice receives the same command and the same activations;
// each has its own weight/scale streams (on F2, slice s <- HBM pseudo-channel s).
// A command covers N = NSlice * RowInterleave * nrowblk rows.
//
// Output (y_*) carries {index, value}:
//   stream mode (cmd_argmax_i = 0): all N rows, merged back into global row order.
//   argmax mode (cmd_argmax_i = 1): one transfer, the row with the largest value
//     under bpu_compute_pkg::f32_order_key, ties to the smallest index
//     (bpuref.qmv.argmax_ref). Used for greedy sampling on the LM head, so the
//     248,320 logits never leave the engine.
//
// Commands are serialized: cmd_ready_o rises once the previous command's results
// have all been delivered. Activations may be written only while cmd_ready_o is high.
// The merge emits at most one row per cycle, which keeps up with the slices
// whenever (K/64) * chunks >= NSlice (every Qwen3.5 shape at the fpga config).
module bpu_qmv_array #(
  parameter int unsigned NSlice        = 32,
  parameter int unsigned Lanes         = 64,
  parameter int unsigned RowInterleave = 4,
  parameter int unsigned MaxK          = 6144,
  parameter int unsigned RowBlkW       = 16,
  parameter int unsigned IdxW          = 24,     // global row index width
  parameter int unsigned OutFifoDepth  = 2 * RowInterleave,
  parameter bit          ProdReg       = 1'b1,
  parameter int unsigned TreeRegEvery  = 2,
  parameter bit          I2fReg        = 1'b1,
  parameter logic [2:0]  MulPipe       = 3'b111,
  parameter logic [2:0]  AddPipe       = 3'b111,
  parameter bit          EnPerf        = 1'b1
) (
  input  logic                                clk_i,
  input  logic                                rst_ni,

  // Activations, broadcast to every slice
  input  logic                                x_we_i,
  input  logic [$clog2(MaxK/Lanes)-1:0]       x_waddr_i,
  input  logic [Lanes*8-1:0]                  x_wdata_i,
  input  logic                                xs_we_i,
  input  logic [$clog2(MaxK/64)-1:0]          xs_waddr_i,
  input  logic [15:0]                         xs_wdata_i,

  // Command
  input  logic                                cmd_valid_i,
  output logic                                cmd_ready_o,
  input  logic                                cmd_wfmt_i,
  input  logic [$clog2(MaxK/64+1)-1:0]        cmd_ngroups_i,  // K / 64
  input  logic [RowBlkW-1:0]                  cmd_nrowblk_i,  // N / (NSlice * RowInterleave)
  input  logic                                cmd_argmax_i,

  // Per-slice weight and scale streams (slice s owns bit s / field s)
  input  logic [NSlice-1:0]                   w_valid_i,
  output logic [NSlice-1:0]                   w_ready_o,
  input  logic [NSlice*Lanes*4-1:0]           w_data_i,
  input  logic [NSlice-1:0]                   ws_valid_i,
  output logic [NSlice-1:0]                   ws_ready_o,
  input  logic [NSlice*16-1:0]                ws_data_i,

  // Results
  output logic                                y_valid_o,
  input  logic                                y_ready_i,
  output logic [31:0]                         y_data_o,
  output logic [IdxW-1:0]                     y_index_o,

  output logic                                busy_o,

  // Status (OR over slices) and per-slice performance counters
  input  logic                                status_clr_i,
  output logic                                err_cmd_o,
  output logic                                flag_nan_o,
  output logic                                flag_inf_o,
  output logic [NSlice*32-1:0]                perf_beats_o,
  output logic [NSlice*32-1:0]                perf_stall_w_o,
  output logic [NSlice*32-1:0]                perf_stall_ws_o,
  output logic [NSlice*32-1:0]                perf_stall_out_o
);

  import bpu_compute_pkg::*;

  localparam int unsigned GW   = $clog2(MaxK / 64 + 1);
  localparam int unsigned SW   = (NSlice > 1) ? $clog2(NSlice) : 1;
  localparam int unsigned LocW = RowBlkW + $clog2(RowInterleave) + 1;   // local row count

  typedef enum logic [2:0] {
    OutIdle, OutStream, OutCollect, OutReduce, OutEmit
  } out_state_e;

  out_state_e state_q;

  // ---------------------------------------------------------------------------
  // Slices
  // ---------------------------------------------------------------------------
  logic [NSlice-1:0]       s_cmd_ready, s_y_valid, s_y_ready, s_busy;
  logic [NSlice-1:0]       s_err, s_nan, s_inf;
  logic [NSlice*32-1:0]    s_y_data;
  logic                    cmd_fire, cmd_legal;

  assign cmd_ready_o = (&s_cmd_ready) && (state_q == OutIdle);
  assign cmd_fire    = cmd_valid_i && cmd_ready_o;
  assign cmd_legal   = (cmd_ngroups_i != '0) && (cmd_ngroups_i <= GW'(MaxK / 64))
                    && (cmd_nrowblk_i != '0);

  for (genvar s = 0; s < NSlice; s++) begin : g_slice
    bpu_qmv_slice #(
      .Lanes(Lanes), .RowInterleave(RowInterleave), .MaxK(MaxK), .RowBlkW(RowBlkW),
      .OutFifoDepth(OutFifoDepth), .ProdReg(ProdReg), .TreeRegEvery(TreeRegEvery),
      .I2fReg(I2fReg), .MulPipe(MulPipe), .AddPipe(AddPipe), .EnPerf(EnPerf)
    ) u_slice (
      .clk_i, .rst_ni,
      .x_we_i, .x_waddr_i, .x_wdata_i, .xs_we_i, .xs_waddr_i, .xs_wdata_i,
      .cmd_valid_i(cmd_fire), .cmd_ready_o(s_cmd_ready[s]), .cmd_wfmt_i,
      .cmd_ngroups_i, .cmd_nrowblk_i,
      .w_valid_i(w_valid_i[s]), .w_ready_o(w_ready_o[s]), .w_data_i(w_data_i[s*Lanes*4 +: Lanes*4]),
      .ws_valid_i(ws_valid_i[s]), .ws_ready_o(ws_ready_o[s]), .ws_data_i(ws_data_i[s*16 +: 16]),
      .y_valid_o(s_y_valid[s]), .y_ready_i(s_y_ready[s]), .y_data_o(s_y_data[s*32 +: 32]),
      .busy_o(s_busy[s]),
      .status_clr_i, .err_cmd_o(s_err[s]), .flag_nan_o(s_nan[s]), .flag_inf_o(s_inf[s]),
      .perf_beats_o(perf_beats_o[s*32 +: 32]), .perf_stall_w_o(perf_stall_w_o[s*32 +: 32]),
      .perf_stall_ws_o(perf_stall_ws_o[s*32 +: 32]), .perf_stall_out_o(perf_stall_out_o[s*32 +: 32])
    );
  end

  assign err_cmd_o  = |s_err;
  assign flag_nan_o = |s_nan;
  assign flag_inf_o = |s_inf;
  assign busy_o     = (|s_busy) || (state_q != OutIdle);

  // ---------------------------------------------------------------------------
  // Output: in-order merge (stream mode) or argmax (collect, reduce, emit)
  // ---------------------------------------------------------------------------
  logic [IdxW-1:0]     idx_q, last_idx_q;   // next global row / last row of the command
  logic [SW-1:0]       cur_q;                // slice holding the next global row
  logic [LocW-1:0]     rows_per_slice_q;
  logic [SW-1:0]       red_q;                // argmax reduction cursor
  logic                have_q;
  logic [31:0]         best_key_q, best_val_q;
  logic [IdxW-1:0]     best_idx_q;
  logic                y_pop;

  assign y_pop = y_valid_o && y_ready_i;

  // Per-slice running argmax over that slice's rows (local order = global order).
  logic [NSlice-1:0]        a_done;
  logic [NSlice*32-1:0]     a_key, a_val;
  logic [NSlice*LocW-1:0]   a_lidx;

  for (genvar s = 0; s < NSlice; s++) begin : g_amax
    logic [LocW-1:0] cnt_q, lidx_q;
    logic [31:0]     key_q, val_q, key_in;
    logic            pop;

    assign key_in = f32_order_key(s_y_data[s*32 +: 32]);
    assign pop    = s_y_valid[s] && s_y_ready[s] && (state_q == OutCollect);

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        cnt_q <= '0;
      end else if (cmd_fire) begin
        cnt_q <= '0;
      end else if (pop) begin
        cnt_q <= cnt_q + 1'b1;
      end
    end

    // Strictly greater keeps the first (smallest-index) maximum.
    always_ff @(posedge clk_i) begin
      if (pop && (cnt_q == '0 || key_in > key_q)) begin
        key_q  <= key_in;
        val_q  <= s_y_data[s*32 +: 32];
        lidx_q <= cnt_q;
      end
    end

    assign a_done[s]                 = (cnt_q == rows_per_slice_q);
    assign a_key[s*32 +: 32]         = key_q;
    assign a_val[s*32 +: 32]         = val_q;
    assign a_lidx[s*LocW +: LocW]    = lidx_q;
  end

  // Slice ready: the merge takes the current slice's head; argmax takes everything.
  always_comb begin
    s_y_ready = '0;
    if (state_q == OutStream)  s_y_ready[cur_q] = y_ready_i;
    if (state_q == OutCollect) s_y_ready = ~a_done;
  end

  // Reduction candidate: slice red_q's best, as a global row index.
  logic [31:0]     cand_key, cand_val;
  logic [IdxW-1:0] cand_idx;
  logic            cand_wins;

  assign cand_key  = a_key[red_q*32 +: 32];
  assign cand_val  = a_val[red_q*32 +: 32];
  assign cand_idx  = IdxW'(a_lidx[red_q*LocW +: LocW]) * IdxW'(NSlice) + IdxW'(red_q);
  assign cand_wins = !have_q || (cand_key > best_key_q)
                  || (cand_key == best_key_q && cand_idx < best_idx_q);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q          <= OutIdle;
      idx_q            <= '0;
      last_idx_q       <= '0;
      cur_q            <= '0;
      rows_per_slice_q <= '0;
      red_q            <= '0;
      have_q           <= 1'b0;
    end else begin
      unique case (state_q)
        OutIdle: begin
          if (cmd_fire && cmd_legal) begin
            state_q          <= cmd_argmax_i ? OutCollect : OutStream;
            idx_q            <= '0;
            cur_q            <= '0;
            last_idx_q       <= IdxW'(cmd_nrowblk_i) * IdxW'(NSlice * RowInterleave) - 1'b1;
            rows_per_slice_q <= LocW'(cmd_nrowblk_i) * LocW'(RowInterleave);
          end
        end
        OutStream: begin
          if (y_pop) begin
            idx_q <= idx_q + 1'b1;
            cur_q <= (cur_q == SW'(NSlice - 1)) ? '0 : cur_q + 1'b1;
            if (idx_q == last_idx_q) state_q <= OutIdle;
          end
        end
        OutCollect: begin
          if (&a_done) begin
            state_q <= OutReduce;
            red_q   <= '0;
            have_q  <= 1'b0;
          end
        end
        OutReduce: begin
          have_q <= 1'b1;
          if (red_q == SW'(NSlice - 1)) state_q <= OutEmit;
          else                          red_q   <= red_q + 1'b1;
        end
        OutEmit: begin
          if (y_pop) state_q <= OutIdle;
        end
        default: state_q <= OutIdle;
      endcase
    end
  end

  always_ff @(posedge clk_i) begin
    if (state_q == OutReduce && cand_wins) begin
      best_key_q <= cand_key;
      best_val_q <= cand_val;
      best_idx_q <= cand_idx;
    end
  end

  assign y_valid_o = (state_q == OutStream) ? s_y_valid[cur_q] : (state_q == OutEmit);
  assign y_data_o  = (state_q == OutStream) ? s_y_data[cur_q*32 +: 32] : best_val_q;
  assign y_index_o = (state_q == OutStream) ? idx_q : best_idx_q;

`ifndef SYNTHESIS
  initial begin
    if (NSlice < 1) $fatal(1, "bpu_qmv_array: NSlice must be >= 1");
    if ((NSlice * RowInterleave) << RowBlkW > (1 << IdxW))
      $fatal(1, "bpu_qmv_array: IdxW=%0d too small for NSlice*R*2^RowBlkW rows", IdxW);
  end
`endif

endmodule
