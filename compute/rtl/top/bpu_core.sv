// BPU core: the command sequencer and scoreboard, the vector unit (bpu_fvu) and the
// matrix unit (bpu_qmv_engine), all working out of one banked shared SRAM - the
// compute and SRAM half of the BPU block diagram.
//
// Outside the core: the control SoC (produces descriptors, watches the scoreboard)
// and the memory manager (HBM <-> SRAM and SRAM <-> SRAM transfers, weight streams).
// The memory manager receives its commands from the sequencer (unit 2) and has its
// own read and write port on the shared SRAM; it also answers the matrix unit's
// weight requests (wreq_*) with the per-slice weight streams (w_*, ws_*).
//
// Shared SRAM ports: reads  [vector x FRdPorts | matrix | memory manager],
//                    writes [vector | matrix | memory manager].
// Descriptor layout and encodings: bpu_isa_pkg. Element addresses throughout.
module bpu_core #(
  // Shared SRAM: NBanks x BankWords words of VLanes fp32 elements
  parameter int unsigned NBanks        = 4,
  parameter int unsigned BankWords     = 16384,
  parameter bit          SOutReg       = 1'b0,
  parameter bit          SHash         = 1'b1,
  // Vector unit
  parameter int unsigned VLanes        = 2,
  parameter logic [2:0]  FMulPipe      = 3'b010,
  parameter logic [2:0]  FAddPipe      = 3'b010,
  parameter logic [4:0]  FSfuPipe      = 5'b01010,
  parameter int unsigned RedFifoDepth  = 8,
  parameter int unsigned FSfuLanes     = VLanes,
  parameter int unsigned FRdPorts      = 1,
  parameter int unsigned FNSlot        = 4,
  parameter int unsigned FWbDepth      = 8,
  parameter int unsigned FAccDepth     = 8,
  // Matrix unit
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
  // Sequencer
  parameter int unsigned NTags         = 16,
  parameter int unsigned QDepth        = 2,
  parameter int unsigned MemBodyW      = 432        // body bits the memory manager uses
) (
  input  logic                                clk_i,
  input  logic                                rst_ni,

  // Descriptors from the control SoC
  input  logic                                desc_valid_i,
  output logic                                desc_ready_o,
  input  logic [1:0]                          desc_unit_i,
  input  logic [$clog2(NTags)-1:0]            desc_tag_i,
  input  logic [NTags-1:0]                    desc_wait_i,
  input  logic [bpu_isa_pkg::BodyW-1:0]       desc_body_i,
  // Scoreboard
  output logic [NTags-1:0]                    pending_o,
  output logic [NTags-1:0]                    err_o,
  output logic [NTags-1:0]                    cpl_o,
  output logic                                busy_o,

  // Memory manager: commands and completion
  output logic                                mm_cmd_valid_o,
  input  logic                                mm_cmd_ready_i,
  output logic [bpu_isa_pkg::BodyW-1:0]       mm_cmd_body_o,
  input  logic                                mm_done_i,
  input  logic                                mm_err_i,
  // Memory manager: shared-SRAM ports (word addresses)
  input  logic                                mm_rd_valid_i,
  output logic                                mm_rd_ready_o,
  input  logic [31-$clog2(VLanes):0]          mm_rd_addr_i,
  output logic                                mm_rd_rvalid_o,
  output logic [VLanes*32-1:0]                mm_rd_rdata_o,
  input  logic                                mm_wr_valid_i,
  output logic                                mm_wr_ready_o,
  input  logic [31-$clog2(VLanes):0]          mm_wr_addr_i,
  input  logic [VLanes-1:0]                   mm_wr_mask_i,
  input  logic [VLanes*32-1:0]                mm_wr_data_i,

  // Weight request to the memory manager, then per-slice weight/scale streams
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

  output logic                                flag_nan_o,     // matrix unit, current command
  output logic                                flag_inf_o,
  output logic [31:0]                         perf_vec_items_o,
  output logic [31:0]                         perf_vec_busy_o,
  output logic [31:0]                         perf_sram_rd_wait_o,
  output logic [31:0]                         perf_sram_wr_wait_o
);

  import bpu_isa_pkg::*;

  localparam int unsigned V   = VLanes;
  localparam int unsigned MAW = 32 - $clog2(V);
  localparam int unsigned NP  = FRdPorts;
  localparam int unsigned NRd = NP + 2;
  localparam int unsigned NWr = 3;

  // ---------------------------------------------------------------------------
  // Sequencer and scoreboard
  // ---------------------------------------------------------------------------
  logic [2:0]         iss_valid, iss_ready, done, done_err;
  logic [3*BodyW-1:0] iss_body;

  bpu_cmd_seq #(.NTags(NTags), .QDepth(QDepth), .BodyW(BodyW), .VecBodyW(VecBodyW),
                .MatBodyW(MatBodyW), .MemBodyW(MemBodyW)) u_seq (
    .clk_i, .rst_ni,
    .desc_valid_i, .desc_ready_o, .desc_unit_i, .desc_tag_i, .desc_wait_i, .desc_body_i,
    .iss_valid_o(iss_valid), .iss_ready_i(iss_ready), .iss_body_o(iss_body),
    .done_i(done), .done_err_i(done_err),
    .pending_o, .err_o, .cpl_o, .busy_o, .perf_accept_wait_o());

  logic [BodyW-1:0] vb, mb;
  assign vb = iss_body[0 +: BodyW];
  assign mb = iss_body[BodyW +: BodyW];
  function automatic logic [31:0] vfield(input logic [BodyW-1:0] b, input int i);
    return b[VbAddr + 32*i +: 32];
  endfunction

  // ---------------------------------------------------------------------------
  // Shared SRAM wiring
  // ---------------------------------------------------------------------------
  logic [NRd-1:0]        rv, rr, rrv, roob;
  logic [NRd*MAW-1:0]    ra;
  logic [NRd*V*32-1:0]   rd;
  logic [NWr-1:0]        wv, wr, woob;
  logic [NWr*MAW-1:0]    wa;
  logic [NWr*V-1:0]      wm;
  logic [NWr*V*32-1:0]   wd;

  bpu_sram_shared #(
    .NBanks(NBanks), .BankWords(BankWords), .Lanes(V), .NRd(NRd), .NWr(NWr), .AW(MAW),
    .OutReg(SOutReg), .Hash(SHash)
  ) u_sram (
    .clk_i, .rst_ni,
    .rd_valid_i(rv), .rd_ready_o(rr), .rd_addr_i(ra), .rd_rvalid_o(rrv), .rd_rdata_o(rd),
    .rd_oob_o(roob),
    .wr_valid_i(wv), .wr_ready_o(wr), .wr_addr_i(wa), .wr_mask_i(wm), .wr_data_i(wd),
    .wr_oob_o(woob),
    .perf_rd_wait_o(perf_sram_rd_wait_o), .perf_wr_wait_o(perf_sram_wr_wait_o));

  // Memory manager ports (read NP+1, write 2)
  assign rv[NP+1]                    = mm_rd_valid_i;
  assign ra[(NP+1)*MAW +: MAW]       = mm_rd_addr_i;
  assign mm_rd_ready_o               = rr[NP+1];
  assign mm_rd_rvalid_o              = rrv[NP+1];
  assign mm_rd_rdata_o               = rd[(NP+1)*V*32 +: V*32];
  assign wv[2]                       = mm_wr_valid_i;
  assign wa[2*MAW +: MAW]            = mm_wr_addr_i;
  assign wm[2*V +: V]                = mm_wr_mask_i;
  assign wd[2*V*32 +: V*32]          = mm_wr_data_i;
  assign mm_wr_ready_o               = wr[2];
  logic unused_mm_oob;
  assign unused_mm_oob = roob[NP+1] ^ woob[2];   // the memory manager checks its own ranges

  // ---------------------------------------------------------------------------
  // Vector unit
  // ---------------------------------------------------------------------------
  bpu_fvu #(
    .VLanes(V), .MulPipe(FMulPipe), .AddPipe(FAddPipe), .SfuPipe(FSfuPipe),
    .RedFifoDepth(RedFifoDepth), .SfuLanes(FSfuLanes), .RdPorts(NP), .NSlot(FNSlot),
    .WbDepth(FWbDepth), .AccDepth(FAccDepth), .EnPerf(1'b1)
  ) u_vec (
    .clk_i, .rst_ni,
    .cmd_valid_i(iss_valid[0]), .cmd_ready_o(iss_ready[0]),
    .cmd_op_i(vb[VbOp +: 5]), .cmd_func_i(vb[VbFunc +: 3]), .cmd_half_log2_i(vb[VbHalf +: 5]),
    .cmd_rows_i(vb[VbRows +: 16]), .cmd_cols_i(vb[VbCols +: 16]),
    .cmd_d_i(vfield(vb, 0)), .cmd_a_i(vfield(vb, 1)), .cmd_b_i(vfield(vb, 2)),
    .cmd_c_i(vfield(vb, 3)), .cmd_s_i(vfield(vb, 4)), .cmd_t_i(vfield(vb, 5)),
    .cmd_ds_i(vfield(vb, 6)), .cmd_as_i(vfield(vb, 7)), .cmd_bs_i(vfield(vb, 8)),
    .cmd_cs_i(vfield(vb, 9)), .cmd_ss_i(vfield(vb, 10)), .cmd_ts_i(vfield(vb, 11)),
    .done_o(done[0]), .err_o(done_err[0]),
    .mrd_valid_o(rv[NP-1:0]), .mrd_ready_i(rr[NP-1:0]), .mrd_addr_o(ra[NP*MAW-1:0]),
    .mrd_rvalid_i(rrv[NP-1:0]), .mrd_rdata_i(rd[NP*V*32-1:0]), .mrd_oob_i(roob[NP-1:0]),
    .mwr_valid_o(wv[0]), .mwr_ready_i(wr[0]), .mwr_addr_o(wa[0 +: MAW]), .mwr_mask_o(wm[0 +: V]),
    .mwr_data_o(wd[0 +: V*32]), .mwr_oob_i(woob[0]),
    .busy_o(), .perf_items_o(perf_vec_items_o), .perf_busy_o(perf_vec_busy_o),
    .perf_stall_rd_o(), .perf_stall_wr_o());

  // ---------------------------------------------------------------------------
  // Matrix unit
  // ---------------------------------------------------------------------------
  logic [MbNW-1:0] m_n;
  logic            m_n_fits;
  assign m_n      = mb[MbN +: MbNW];
  assign m_n_fits = (32'(m_n) >> IdxW) == '0;

  bpu_qmv_engine #(
    .NSlice(NSlice), .Lanes(Lanes), .RowInterleave(RowInterleave), .MaxK(MaxK),
    .RowBlkW(RowBlkW), .IdxW(IdxW), .ProdReg(QProdReg), .TreeRegEvery(QTreeRegEvery),
    .I2fReg(QI2fReg), .MulPipe(QMulPipe), .AddPipe(QAddPipe), .VLanes(V), .LdDepth(4)
  ) u_mat (
    .clk_i, .rst_ni,
    .cmd_valid_i(iss_valid[1]), .cmd_ready_o(iss_ready[1]),
    .cmd_wid_i(mb[MbWid +: 16]), .cmd_wfmt_i(mb[MbWfmt]), .cmd_argmax_i(mb[MbArgmax]),
    // a row count wider than the engine's index makes the command illegal (k = 0)
    .cmd_k_i(m_n_fits ? mb[MbK +: 16] : 16'd0), .cmd_n_i(IdxW'(m_n)),
    .cmd_x_i(mb[MbX +: 32]), .cmd_xs_i(mb[MbXs +: 32]), .cmd_y_i(mb[MbY +: 32]),
    .done_o(done[1]), .err_o(done_err[1]),
    .mrd_valid_o(rv[NP]), .mrd_ready_i(rr[NP]), .mrd_addr_o(ra[NP*MAW +: MAW]),
    .mrd_rvalid_i(rrv[NP]), .mrd_rdata_i(rd[NP*V*32 +: V*32]), .mrd_oob_i(roob[NP]),
    .mwr_valid_o(wv[1]), .mwr_ready_i(wr[1]), .mwr_addr_o(wa[MAW +: MAW]), .mwr_mask_o(wm[V +: V]),
    .mwr_data_o(wd[V*32 +: V*32]), .mwr_oob_i(woob[1]),
    .wreq_valid_o, .wreq_ready_i, .wreq_id_o, .wreq_nrowblk_o, .wreq_ngroups_o, .wreq_wfmt_o,
    .w_valid_i, .w_ready_o, .w_data_i, .ws_valid_i, .ws_ready_o, .ws_data_i,
    .busy_o(), .flag_nan_o, .flag_inf_o);

  // ---------------------------------------------------------------------------
  // Memory manager
  // ---------------------------------------------------------------------------
  assign mm_cmd_valid_o = iss_valid[2];
  assign iss_ready[2]   = mm_cmd_ready_i;
  assign mm_cmd_body_o  = iss_body[2*BodyW +: BodyW];
  assign done[2]        = mm_done_i;
  assign done_err[2]    = mm_err_i;

endmodule
