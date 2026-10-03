// FP32 vector unit: executes bpuref.fvu ops on operands in the shared SRAM,
// bit-exact with bpuref.fvu.execute for every VLanes.
//
// Memory: element e lives in word e / VLanes of the shared SRAM (bpu_sram_shared).
// The unit has RdPorts operand read ports and one result write port, all
// request/response clients: requests may wait any number of cycles, read
// responses come back in request order. Command addresses and strides are 32-bit
// element addresses. Ops are 2-D (rows x cols) with per-row strided operands (see
// bpuref/fvu.py): vector operands and their strides are multiples of 64 elements,
// scalars are any element.
//
// Pipeline:
//   walker      walks (row, word) items and allocates one operand-collector slot
//               per item, with the item's addresses and which operands it needs
//               (row scalar s, group scalar t, a, b, c);
//   collector   each read port fetches its class of operands (port 0: s, t, a;
//               port 1: b; port 2: c; one port serves all) for the slots in
//               allocation order; responses land in the slot;
//   launch      the oldest slot launches once all its operands are in, the write
//               buffer has room for its result, and (reductions) the reduction FIFO
//               has a credit. A VSFU word with a shared SFU bank launches once per
//               lane group;
//   lanes       fixed latency, never stall; element-wise results enter the write
//               buffer, row reductions enter bpu_fvu_reduce;
//   write       the write buffer drains through the write port.
// VVECMAT  d = (s[r] * a[r,:]) + (r == 0 ? +0 : d)  row after row: when a row fits
// the AccDepth-word forwarding FIFO, row r+1 takes row r's results from that FIFO
// and only the last row is written. Longer rows read the accumulator back from
// the SRAM, and a read of word w of row r+1 waits until the write of word w of row
// r has been accepted (counted, not timed), so any memory timing is safe.
//
// One op at a time. done_o pulses when the op's last write has been accepted
// (so the result is visible to any later reader); err_o qualifies it: an illegal
// descriptor (consumed without running) or an out-of-range address during the op.
module bpu_fvu #(
  parameter int unsigned VLanes       = 4,         // power of two, 2..64
  parameter logic [2:0]  MulPipe      = 3'b111,
  parameter logic [2:0]  AddPipe      = 3'b111,
  parameter logic [4:0]  SfuPipe      = 5'b11111,
  parameter int unsigned RedFifoDepth = 8,
  parameter int unsigned SfuLanes     = VLanes,    // SFUs shared by the lanes (power of two <= VLanes)
  parameter int unsigned RdPorts      = 1,         // operand read ports, 1..3
  parameter int unsigned NSlot        = 4,         // operand-collector slots, power of two 2..8
  parameter int unsigned WbDepth      = 8,         // write buffer entries
  parameter int unsigned AccDepth     = 8,         // VVECMAT forwarding FIFO words (0: none)
  parameter bit          EnPerf       = 1'b1
) (
  input  logic                        clk_i,
  input  logic                        rst_ni,

  // Command (one op)
  input  logic                        cmd_valid_i,
  output logic                        cmd_ready_o,
  input  logic [4:0]                  cmd_op_i,
  input  logic [2:0]                  cmd_func_i,
  input  logic [4:0]                  cmd_half_log2_i,
  input  logic [15:0]                 cmd_rows_i,
  input  logic [15:0]                 cmd_cols_i,
  input  logic [31:0]                 cmd_d_i, cmd_a_i, cmd_b_i, cmd_c_i, cmd_s_i, cmd_t_i,
  input  logic [31:0]                 cmd_ds_i, cmd_as_i, cmd_bs_i, cmd_cs_i, cmd_ss_i, cmd_ts_i,
  output logic                        done_o,
  output logic                        err_o,

  // Shared SRAM: operand reads
  output logic [RdPorts-1:0]          mrd_valid_o,
  input  logic [RdPorts-1:0]          mrd_ready_i,
  output logic [RdPorts*(32-$clog2(VLanes))-1:0] mrd_addr_o,
  input  logic [RdPorts-1:0]          mrd_rvalid_i,
  input  logic [RdPorts*VLanes*32-1:0] mrd_rdata_i,
  input  logic [RdPorts-1:0]          mrd_oob_i,
  // Shared SRAM: result writes
  output logic                        mwr_valid_o,
  input  logic                        mwr_ready_i,
  output logic [31-$clog2(VLanes):0]  mwr_addr_o,
  output logic [VLanes-1:0]           mwr_mask_o,
  output logic [VLanes*32-1:0]        mwr_data_o,
  input  logic                        mwr_oob_i,

  output logic                        busy_o,
  output logic [31:0]                 perf_items_o,      // lane launches
  output logic [31:0]                 perf_busy_o,       // cycles with an op running
  output logic [31:0]                 perf_stall_rd_o,   // oldest item waiting for operands
  output logic [31:0]                 perf_stall_wr_o    // oldest item waiting for write-buffer room
);

  import bpu_compute_pkg::*;
  import bpu_isa_pkg::*;

  localparam int unsigned V    = VLanes;
  localparam int unsigned LW   = $clog2(V);
  localparam int unsigned MAW  = 32 - LW;                  // SRAM word-address width
  localparam int unsigned NP   = RdPorts;
  localparam int unsigned SW   = $clog2(NSlot);
  localparam int unsigned Lm   = pipe3_latency(MulPipe);
  localparam int unsigned La   = pipe3_latency(AddPipe);
  localparam int unsigned Ls   = int'(SfuPipe[0]) + int'(SfuPipe[1]) + int'(SfuPipe[2])
                               + int'(SfuPipe[3]) + int'(SfuPipe[4]);
  localparam int unsigned Ltot = (Lm + La > Ls) ? Lm + La : Ls;
  localparam logic [4:0]  LopAbs = 5'd14;
  localparam int unsigned NSub   = V / SfuLanes;             // VSFU sub-items per word
  localparam int unsigned KW     = (NSub > 1) ? $clog2(NSub) : 1;
  localparam int unsigned TagW   = SW + 3 + LW;              // read tag: slot, operand, lane
  localparam int unsigned TagDepth = 3 * NSlot;              // reads one port can have in flight

  // Operand indices
  localparam int unsigned OpS = 0, OpT = 1, OpA = 2, OpB = 3, OpC = 4;

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
  // Operands each read port serves.
  function automatic logic [4:0] port_class(input int p);
    if (NP == 1) return 5'b11111;
    if (p == 0)  return 5'b00111;                    // s, t, a
    if (NP == 2) return 5'b11000;                    // b, c
    return (p == 1) ? 5'b01000 : 5'b10000;           // b | c
  endfunction

  // ---------------------------------------------------------------------------
  // Command accept and legality
  // ---------------------------------------------------------------------------
  typedef enum logic [1:0] {SIdle, SRun} state_e;
  state_e st_q;
  logic   op_done;

  logic cmd_fire, cmd_legal, cmd_red, cmd_ew;
  logic [16:0] p2, cmd_words;

  assign cmd_ready_o = (st_q == SIdle);
  assign cmd_fire    = cmd_valid_i && cmd_ready_o;
  assign cmd_red     = op_red(cmd_op_i);
  assign cmd_ew      = !cmd_red;                   // element-wise and VVECMAT write vectors
  assign p2          = pow2_ceil(cmd_cols_i);
  assign cmd_words   = 17'((32'(cmd_cols_i) + V - 1) >> LW);

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
    if (cmd_op_i == FvuVsfu && cmd_func_i > 3'(SfuLog2)) cmd_legal = 1'b0;
  end

  // Descriptor registers
  logic [4:0]    op_q, lop_q;
  logic [2:0]    func_q;
  logic [15:0]   rows_q, cols_q;
  logic [16:0]   half_q, wreal_q, wrow_q;
  logic [31:0]   ds_q, as_q, bs_q, cs_q, ss_q, ts_q;
  logic          red_q, vm_q, max_q, ub_q, uc_q, us_q, ut_q, sub_q, fwd_q;

  // ---------------------------------------------------------------------------
  // Walker: one item (row r, word w) per cycle into a free collector slot
  // ---------------------------------------------------------------------------
  logic [15:0]   r_q;
  logic [16:0]   w_q;
  logic [31:0]   n_q;                                       // item index within the op
  logic [31:0]   rd_q, ra_q, rb_q, rc_q, rs_q, rt_q;        // row bases
  logic          walk_done_q;

  logic [31:0]   eoff;
  logic          pad, last_w, last_r, row_start;
  logic [4:0]    w_need;
  logic [V-1:0]  w_mask;
  logic          alloc;
  logic [SW:0]   cnt_q;                                     // allocated slots
  logic [SW-1:0] head_q, tail_q;

  assign eoff      = 32'(w_q) << LW;
  assign pad       = red_q && (w_q >= wreal_q);
  assign last_w    = (w_q == wrow_q - 1'b1);
  assign last_r    = (r_q == rows_q - 1'b1);
  assign row_start = (w_q == '0);

  assign w_need[OpS] = us_q && row_start;
  assign w_need[OpT] = ut_q && (eoff[5:0] == '0);              // a new 64-element group
  assign w_need[OpA] = !pad;
  assign w_need[OpB] = (ub_q && !pad) || (vm_q && !fwd_q && r_q != '0);
  assign w_need[OpC] = uc_q && !pad;
  for (genvar l = 0; l < V; l++) begin : g_wmask
    assign w_mask[l] = !pad && ((eoff + 32'(l)) < 32'(cols_q));
  end

  assign alloc = (st_q == SRun) && !walk_done_q && (cnt_q != (SW+1)'(NSlot));

  // ---------------------------------------------------------------------------
  // Operand-collector slots
  // ---------------------------------------------------------------------------
  logic [NSlot-1:0]             sv_q;                       // allocated
  logic [NSlot-1:0][4:0]        need_q, iss_q, got_q;
  logic [NSlot-1:0][4:0][31:0]  el_q;                       // operand element addresses
  logic [NSlot-1:0][V-1:0]      smask_q;
  logic [NSlot-1:0][31:0]       sdest_q, accw_q;            // destination; accumulator wait count
  logic [NSlot-1:0]             sfirst_q, slast_q, szb_q, sfpop_q, spush_q, swr_q;
  logic [NSlot-1:0][31:0]       ds_dat_q, dt_dat_q;
  logic [NSlot-1:0][V*32-1:0]   da_dat_q, db_dat_q, dc_dat_q;

  // Writes accepted during this op (VVECMAT accumulator ordering).
  logic [31:0] wr_cnt_q;

  // ---------------------------------------------------------------------------
  // Read ports: oldest slot with a missing operand of the port's class
  // ---------------------------------------------------------------------------
  logic [NP-1:0]          pv, pacc;
  logic [NP-1:0][SW-1:0]  pslot;
  logic [NP-1:0][2:0]     popnd;
  logic [NP-1:0][31:0]    pel;

  always_comb begin
    for (int p = 0; p < NP; p++) begin
      logic       found;
      logic [4:0] miss;
      int unsigned j;
      found    = 1'b0;
      pslot[p] = '0;
      popnd[p] = '0;
      pel[p]   = '0;
      for (int i = 0; i < NSlot; i++) begin
        j    = (int'(head_q) + i) % NSlot;
        miss = need_q[j] & ~iss_q[j] & port_class(p);
        if (!found && sv_q[j] && miss != '0) begin
          found    = 1'b1;
          pslot[p] = SW'(j);
          for (int o = 4; o >= 0; o--) if (miss[o]) popnd[p] = 3'(o);
          pel[p]   = el_q[j][popnd[p]];
        end
      end
      // A VVECMAT accumulator read waits for the previous row's write of that word.
      pacc[p] = found && (popnd[p] == 3'(OpB)) && (accw_q[pslot[p]] != '0)
             && (wr_cnt_q < accw_q[pslot[p]]);
      pv[p]   = found && !pacc[p];
      mrd_valid_o[p]           = pv[p];
      mrd_addr_o[p*MAW +: MAW] = pel[p][31:LW];
    end
  end

  // Read tags: one in-order FIFO per port
  logic [NP-1:0]           tag_push, tag_pop, tag_full, tag_have;
  logic [NP-1:0][TagW-1:0] tag_in, tag_out;
  for (genvar p = 0; p < NP; p++) begin : g_tag
    logic tag_ready;
    assign tag_push[p] = mrd_valid_o[p] && mrd_ready_i[p];
    assign tag_pop[p]  = mrd_rvalid_i[p];
    assign tag_in[p]   = {pslot[p], popnd[p], pel[p][LW-1:0]};
    bpu_fifo #(.Width(TagW), .Depth(TagDepth)) u_tags (
      .clk_i, .rst_ni,
      .in_valid_i(tag_push[p]), .in_ready_o(tag_ready), .in_data_i(tag_in[p]),
      .out_valid_o(tag_have[p]), .out_ready_i(tag_pop[p]), .out_data_o(tag_out[p]),
      .count_o());
    assign tag_full[p] = !tag_ready;
  end

  // ---------------------------------------------------------------------------
  // Launch: the oldest slot, once its operands are in and its result has room
  // ---------------------------------------------------------------------------
  logic [KW-1:0] k_q;                                     // VSFU lane group
  logic          last_k, head_ok, need_wb, need_red, launch;
  logic          acc_have;
  logic [V*32-1:0] acc_head;
  logic [$clog2(WbDepth+1)-1:0]      wb_cred_q;
  logic [$clog2(RedFifoDepth+1)-1:0] red_cred_q;

  assign last_k   = !sub_q || (k_q == KW'(NSub - 1));
  assign head_ok  = sv_q[head_q] && (got_q[head_q] == need_q[head_q])
                 && (!sfpop_q[head_q] || acc_have);
  assign need_wb  = swr_q[head_q];
  assign need_red = red_q;
  assign launch   = (st_q == SRun) && head_ok
                 && (!need_wb || wb_cred_q != '0) && (!need_red || red_cred_q != '0);

  // ---------------------------------------------------------------------------
  // Slot and walker state
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      st_q        <= SIdle;
      walk_done_q <= 1'b0;
      sv_q        <= '0;
      cnt_q       <= '0;
      head_q      <= '0;
      tail_q      <= '0;
      k_q         <= '0;
      r_q         <= '0;
      w_q         <= '0;
      n_q         <= '0;
    end else begin
      if (cmd_fire && cmd_legal) begin
        st_q        <= SRun;
        walk_done_q <= 1'b0;
        r_q         <= '0;
        w_q         <= '0;
        n_q         <= '0;
        k_q         <= '0;
      end
      if (alloc) begin
        sv_q[tail_q] <= 1'b1;
        tail_q       <= tail_q + 1'b1;
        n_q          <= n_q + 1'b1;
        if (!last_w) begin
          w_q <= w_q + 1'b1;
        end else begin
          w_q <= '0;
          if (!last_r) r_q <= r_q + 1'b1;
          else         walk_done_q <= 1'b1;
        end
      end
      if (launch) begin
        if (last_k) begin
          k_q          <= '0;
          sv_q[head_q] <= 1'b0;
          head_q       <= head_q + 1'b1;
        end else begin
          k_q <= k_q + 1'b1;
        end
      end
      cnt_q <= cnt_q + (SW+1)'(alloc) - (SW+1)'(launch && last_k);
      if (st_q == SRun && op_done) st_q <= SIdle;
    end
  end

  // Slot contents: written at allocation, operand flags and data as reads progress.
  always_ff @(posedge clk_i) begin
    if (alloc) begin
      need_q[tail_q]   <= w_need;
      iss_q[tail_q]    <= '0;
      got_q[tail_q]    <= '0;
      el_q[tail_q]     <= {rc_q + eoff, rb_q + eoff,
                           ra_q + ((op_q == FvuVperm && half_q >= 17'(V)) ? (eoff ^ 32'(half_q)) : eoff),
                           rt_q + (eoff >> 6), rs_q};
      smask_q[tail_q]  <= w_mask;
      sdest_q[tail_q]  <= red_q ? rd_q : rd_q + eoff;
      sfirst_q[tail_q] <= row_start;
      slast_q[tail_q]  <= last_w;
      szb_q[tail_q]    <= vm_q && r_q == '0;
      sfpop_q[tail_q]  <= vm_q && fwd_q && r_q != '0;
      spush_q[tail_q]  <= vm_q && fwd_q && !last_r;
      swr_q[tail_q]    <= red_q ? last_w : !(vm_q && fwd_q && !last_r);
      accw_q[tail_q]   <= (vm_q && !fwd_q && r_q != '0) ? n_q - 32'(wrow_q) + 32'd1 : '0;
    end
    for (int p = 0; p < NP; p++) begin
      if (tag_push[p]) iss_q[pslot[p]][popnd[p]] <= 1'b1;
      if (mrd_rvalid_i[p]) begin : rsp
        logic [SW-1:0] js;
        logic [2:0]    o;
        logic [31:0]   lane_val;
        js = tag_out[p][TagW-1 -: SW];
        o  = tag_out[p][TagW-SW-1 -: 3];
        lane_val = mrd_rdata_i[p*V*32 + int'(tag_out[p][LW-1:0]) * 32 +: 32];
        got_q[js][o] <= 1'b1;
        unique case (o)
          3'(OpS): ds_dat_q[js] <= lane_val;
          3'(OpT): dt_dat_q[js] <= lane_val;
          3'(OpA): da_dat_q[js] <= mrd_rdata_i[p*V*32 +: V*32];
          3'(OpB): db_dat_q[js] <= mrd_rdata_i[p*V*32 +: V*32];
          default: dc_dat_q[js] <= mrd_rdata_i[p*V*32 +: V*32];
        endcase
      end
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
      wreal_q <= cmd_words;
      wrow_q  <= cmd_red ? 17'(((p2 < 17'd64) ? 17'd64 : p2) >> LW) : cmd_words;
      red_q   <= cmd_red;
      vm_q    <= (cmd_op_i == FvuVvecmat);
      fwd_q   <= (cmd_op_i == FvuVvecmat) && (AccDepth > 0) && (32'(cmd_words) <= AccDepth);
      max_q   <= (cmd_op_i == FvuRmax) || (cmd_op_i == FvuRamax);
      ub_q    <= op_uses_b(cmd_op_i);
      uc_q    <= op_uses_c(cmd_op_i);
      us_q    <= op_uses_s(cmd_op_i);
      ut_q    <= (cmd_op_i == FvuVmulg);
      sub_q   <= (cmd_op_i == FvuVsfu) && (NSub > 1);
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
    end else if (alloc && last_w) begin
      rd_q <= rd_q + ds_q;
      ra_q <= ra_q + as_q;
      rb_q <= rb_q + bs_q;
      rc_q <= rc_q + cs_q;
      rs_q <= rs_q + ss_q;
      rt_q <= rt_q + ts_q;
    end
  end

  // ---------------------------------------------------------------------------
  // Emit stage: the launched item, registered, into the lanes
  // ---------------------------------------------------------------------------
  localparam int unsigned MW = 1 + 1 + 1 + 1 + 1 + 1 + V + 32;  // red, max, first, last, push, wr, mask, dest
  logic            e_v_q, e_zb_q;
  logic [31:0]     cur_s_q, cur_t_q;          // row / group scalar of the items in flight
  logic [KW-1:0]   e_k_q;
  logic [MW-1:0]   e_m_q;
  logic [31:0]     e_s_q, e_t_q;
  logic [V*32-1:0] e_a_q, e_b_q, e_c_q;
  logic [V-1:0]    l_mask;

  for (genvar l = 0; l < V; l++) begin : g_lmask
    assign l_mask[l] = smask_q[head_q][l] && (!sub_q || (KW'(l / SfuLanes) == k_q));
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) e_v_q <= 1'b0;
    else         e_v_q <= launch;
  end
  always_ff @(posedge clk_i) begin
    if (launch) begin
      e_zb_q <= szb_q[head_q];
      e_k_q  <= k_q;
      e_m_q  <= {red_q, max_q, sfirst_q[head_q], slast_q[head_q] && last_k, spush_q[head_q],
                 swr_q[head_q] && !red_q, l_mask, sdest_q[head_q]};
      // Scalars are fetched by the first item of their row (s) or 64-group (t) and
      // reused by the following items, which launch in order.
      e_s_q  <= need_q[head_q][OpS] ? ds_dat_q[head_q] : cur_s_q;
      e_t_q  <= need_q[head_q][OpT] ? dt_dat_q[head_q] : cur_t_q;
      if (need_q[head_q][OpS]) cur_s_q <= ds_dat_q[head_q];
      if (need_q[head_q][OpT]) cur_t_q <= dt_dat_q[head_q];
      e_b_q  <= sfpop_q[head_q] ? acc_head : db_dat_q[head_q];
      e_c_q  <= dc_dat_q[head_q];
      // VPERM's in-word lane swap (partner distance < VLanes)
      for (int l = 0; l < V; l++)
        e_a_q[l*32 +: 32] <= (op_q == FvuVperm && half_q < 17'(V))
                           ? da_dat_q[head_q][(l ^ int'(half_q[LW-1:0]))*32 +: 32]
                           : da_dat_q[head_q][l*32 +: 32];
    end
  end

  // ---------------------------------------------------------------------------
  // Lanes
  // ---------------------------------------------------------------------------
  // Shared SFU bank (SfuLanes < VLanes): sub-item k feeds lanes k*SfuLanes ..; every
  // lane receives the output of bank unit (lane mod SfuLanes) at the lane's own SFU
  // timing, and write masks keep only the sub-item's lanes.
  logic [V*32-1:0] sf_ext;
  if (NSub > 1) begin : g_sfu_bank
    logic [SfuLanes*32-1:0] bank_out;
    for (genvar j = 0; j < SfuLanes; j++) begin : g_unit
      bpu_sfu #(.PipeMask(SfuPipe)) u_sfu (
        .clk_i, .rst_ni, .valid_i(e_v_q), .func_i(func_q),
        .a_i(e_a_q[(int'(e_k_q) * SfuLanes + j)*32 +: 32]), .valid_o(), .y_o(bank_out[j*32 +: 32]));
    end
    for (genvar l = 0; l < V; l++) begin : g_fan
      assign sf_ext[l*32 +: 32] = bank_out[(l % SfuLanes)*32 +: 32];
    end
  end else begin : g_no_bank
    logic unused_k;
    assign unused_k = ^e_k_q;
    assign sf_ext = '0;
  end

  logic [V*32-1:0] res;
  for (genvar l = 0; l < V; l++) begin : g_lane
    bpu_fvu_lane #(.MulPipe(MulPipe), .AddPipe(AddPipe), .SfuPipe(SfuPipe), .HasSfu(NSub == 1)) u_lane (
      .clk_i, .rst_ni, .valid_i(e_v_q), .lop_i(lop_q), .func_i(func_q), .zero_b_i(e_zb_q),
      .a_i(e_a_q[l*32 +: 32]), .b_i(e_b_q[l*32 +: 32]), .c_i(e_c_q[l*32 +: 32]),
      .s_i(e_s_q), .t_i(e_t_q), .sf_ext_i(sf_ext[l*32 +: 32]), .res_o(res[l*32 +: 32]));
  end

  logic          o_v, o_red, o_max, o_first, o_last, o_push, o_wr;
  logic [V-1:0]  o_mask;
  logic [31:0]   o_dest;

  bpu_delay #(.Width(1), .Depth(Ltot), .Reset(1'b1)) u_ov (
    .clk_i, .rst_ni, .d_i(e_v_q), .q_o(o_v));
  bpu_delay #(.Width(MW), .Depth(Ltot)) u_om (
    .clk_i, .rst_ni, .d_i(e_m_q), .q_o({o_red, o_max, o_first, o_last, o_push, o_wr, o_mask, o_dest}));

  // VVECMAT forwarding FIFO: row r's results, consumed in order by row r + 1.
  // Its occupancy never exceeds one row (each later-row item pops before it pushes).
  logic acc_ready;
  if (AccDepth > 0) begin : g_acc
    bpu_fifo #(.Width(V*32), .Depth(AccDepth)) u_acc (
      .clk_i, .rst_ni,
      .in_valid_i(o_v && o_push), .in_ready_o(acc_ready), .in_data_i(res),
      .out_valid_o(acc_have), .out_ready_i(launch && sfpop_q[head_q]), .out_data_o(acc_head),
      .count_o());
  end else begin : g_no_acc
    assign acc_ready = 1'b1;
    assign acc_have  = 1'b0;
    assign acc_head  = '0;
  end

  // ---------------------------------------------------------------------------
  // Reductions
  // ---------------------------------------------------------------------------
  logic          rw_v, red_busy, red_pop;
  logic [31:0]   rw_addr;
  logic [31:0]   rw_data;

  bpu_fvu_reduce #(.Lanes(V), .AW(32), .AddPipe(AddPipe), .FifoDepth(RedFifoDepth)) u_red (
    .clk_i, .rst_ni,
    .in_valid_i(o_v && o_red), .in_max_i(o_max), .in_first_i(o_first), .in_last_i(o_last),
    .in_dest_i(o_dest), .in_mask_i(o_mask), .in_vals_i(res),
    .pop_o(red_pop), .wr_valid_o(rw_v), .wr_addr_o(rw_addr), .wr_data_o(rw_data),
    .busy_o(red_busy));

  // ---------------------------------------------------------------------------
  // Write buffer -> write port
  // ---------------------------------------------------------------------------
  localparam int unsigned WbW = MAW + V + V*32;
  logic          wb_push, wb_ready, wb_have;
  logic [WbW-1:0] wb_in, wb_out;
  logic [$clog2(WbDepth+1)-1:0] wb_count;

  assign wb_push = (o_v && o_wr) || rw_v;
  assign wb_in   = rw_v ? {rw_addr[31:LW], V'(1) << rw_addr[LW-1:0], {V{rw_data}}}
                        : {o_dest[31:LW], o_mask, res};
  bpu_fifo #(.Width(WbW), .Depth(WbDepth)) u_wb (
    .clk_i, .rst_ni,
    .in_valid_i(wb_push), .in_ready_o(wb_ready), .in_data_i(wb_in),
    .out_valid_o(wb_have), .out_ready_i(mwr_ready_i), .out_data_o(wb_out),
    .count_o(wb_count));
  assign mwr_valid_o = wb_have;
  assign {mwr_addr_o, mwr_mask_o, mwr_data_o} = wb_out;

  logic wr_acc;
  assign wr_acc = mwr_valid_o && mwr_ready_i;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wb_cred_q  <= ($bits(wb_cred_q))'(WbDepth);
      red_cred_q <= ($bits(red_cred_q))'(RedFifoDepth);
      wr_cnt_q   <= '0;
    end else begin
      wb_cred_q  <= wb_cred_q - ($bits(wb_cred_q))'(launch && need_wb) + ($bits(wb_cred_q))'(wr_acc);
      red_cred_q <= red_cred_q - ($bits(red_cred_q))'(launch && need_red) + ($bits(red_cred_q))'(red_pop);
      if (cmd_fire)    wr_cnt_q <= '0;
      else if (wr_acc) wr_cnt_q <= wr_cnt_q + 1'b1;
    end
  end

  // ---------------------------------------------------------------------------
  // Completion
  // ---------------------------------------------------------------------------
  logic [$clog2(Ltot+4)-1:0] inflight_q;          // launched, not yet out of the lanes
  logic op_err_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) inflight_q <= '0;
    else         inflight_q <= inflight_q + ($bits(inflight_q))'(launch) - ($bits(inflight_q))'(o_v);
  end

  assign op_done = walk_done_q && (cnt_q == '0) && (inflight_q == '0) && !red_busy && !rw_v
                && (wb_count == '0);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      op_err_q <= 1'b0;
      done_o   <= 1'b0;
      err_o    <= 1'b0;
    end else begin
      done_o <= 1'b0;
      err_o  <= 1'b0;
      if (cmd_fire) op_err_q <= 1'b0;
      else if ((mrd_oob_i & mrd_valid_o) != '0 || (mwr_oob_i && mwr_valid_o)) op_err_q <= 1'b1;
      if (cmd_fire && !cmd_legal) begin
        done_o <= 1'b1;
        err_o  <= 1'b1;
      end else if (st_q == SRun && op_done) begin
        done_o <= 1'b1;
        err_o  <= op_err_q;
      end
    end
  end

  assign busy_o = (st_q != SIdle);

  if (EnPerf) begin : g_perf
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        perf_items_o    <= '0;
        perf_busy_o     <= '0;
        perf_stall_rd_o <= '0;
        perf_stall_wr_o <= '0;
      end else begin
        perf_items_o    <= perf_items_o + 32'(launch);
        perf_busy_o     <= perf_busy_o + 32'(st_q != SIdle);
        perf_stall_rd_o <= perf_stall_rd_o + 32'(st_q == SRun && sv_q[head_q] && !head_ok);
        perf_stall_wr_o <= perf_stall_wr_o + 32'(st_q == SRun && head_ok && !launch);
      end
    end
  end else begin : g_no_perf
    assign perf_items_o    = '0;
    assign perf_busy_o     = '0;
    assign perf_stall_rd_o = '0;
    assign perf_stall_wr_o = '0;
  end

`ifdef FORMAL
  // Previous-cycle request state, for the "stable until accepted" rule
  logic [NP-1:0]          f_pv_q, f_prdy_q;
  logic [NP-1:0][MAW-1:0] f_paddr_q;
  logic                   f_init_q = 1'b1;
  always_ff @(posedge clk_i) begin
    f_init_q  <= 1'b0;
    f_pv_q    <= mrd_valid_o;
    f_prdy_q  <= mrd_ready_i;
    for (int p = 0; p < NP; p++) f_paddr_q[p] <= mrd_addr_o[p*MAW +: MAW];
  end

  always_comb begin
    if (rst_ni) begin
      // Credits and counts stay in range; buffers never overflow or underflow.
      assert (wb_cred_q <= ($bits(wb_cred_q))'(WbDepth));
      assert (red_cred_q <= ($bits(red_cred_q))'(RedFifoDepth));
      assert (32'(wb_cred_q) + 32'(wb_count) <= WbDepth);
      assert (cnt_q <= (SW+1)'(NSlot));
      assert (!(wb_push && !wb_ready));
      assert (!(o_v && o_push && !acc_ready));
      for (int p = 0; p < NP; p++) begin
        assert (!(tag_push[p] && tag_full[p]));
        assert (!(mrd_rvalid_i[p] && !tag_have[p]));
      end
      // Slot bookkeeping: got <= issued <= needed.
      for (int j = 0; j < NSlot; j++)
        if (sv_q[j]) assert (((got_q[j] & ~iss_q[j]) == '0) && ((iss_q[j] & ~need_q[j]) == '0));
      // A request, once made, holds its address until accepted.
      if (!f_init_q)
        for (int p = 0; p < NP; p++)
          if (f_pv_q[p] && !f_prdy_q[p])
            assert (mrd_valid_o[p] && mrd_addr_o[p*MAW +: MAW] == f_paddr_q[p]);
      // Nothing launches while idle, and the lanes are empty when an op completes.
      if (st_q == SIdle) assert (!launch && !alloc && cnt_q == '0);
    end
  end
`endif

`ifndef SYNTHESIS
  /* verilator lint_off SYNCASYNCNET */
  initial begin
    if (V < 2 || V > 64 || (V & (V - 1)) != 0)
      $fatal(1, "bpu_fvu: VLanes=%0d must be a power of two in [2, 64]", V);
    if (SfuLanes < 1 || SfuLanes > V || (SfuLanes & (SfuLanes - 1)) != 0)
      $fatal(1, "bpu_fvu: SfuLanes=%0d must be a power of two in [1, VLanes]", SfuLanes);
    if (RdPorts < 1 || RdPorts > 3)
      $fatal(1, "bpu_fvu: RdPorts=%0d must be 1, 2 or 3", RdPorts);
    if (NSlot < 2 || NSlot > 8 || (NSlot & (NSlot - 1)) != 0)
      $fatal(1, "bpu_fvu: NSlot=%0d must be 2, 4 or 8", NSlot);
  end
  always @(posedge clk_i) begin
    if (rst_ni) begin
      if (wb_push && !wb_ready) $error("bpu_fvu: write buffer overflow (credit accounting broken)");
      if (o_v && o_push && !acc_ready) $error("bpu_fvu: VVECMAT forwarding FIFO overflow");
      for (int p = 0; p < NP; p++) begin
        if (mrd_rvalid_i[p] && !tag_have[p]) $error("bpu_fvu: read response with no request on port %0d", p);
        if (tag_push[p] && tag_full[p]) $error("bpu_fvu: read tag FIFO overflow on port %0d", p);
      end
    end
  end
  /* verilator lint_on SYNCASYNCNET */
`endif

endmodule
