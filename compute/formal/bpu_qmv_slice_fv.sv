// Formal harness for bpu_qmv_slice. Every input is free except for the
// assumptions below, which encode the documented usage rules. The properties
// themselves live inside the slice (`ifdef FORMAL`), plus output-side checks here.
module bpu_qmv_slice_fv #(
  parameter int unsigned Lanes         = 4,
  parameter int unsigned RowInterleave = 2,
  parameter int unsigned MaxK          = 128,
  parameter int unsigned RowBlkW       = 3,
  parameter int unsigned OutFifoDepth  = 2 * RowInterleave
) (
  input logic                            clk_i,
  input logic                            rst_ni,
  input logic                            x_we_i,
  input logic [$clog2(MaxK/Lanes)-1:0]   x_waddr_i,
  input logic [Lanes*8-1:0]              x_wdata_i,
  input logic                            xs_we_i,
  input logic [$clog2(MaxK/64)-1:0]      xs_waddr_i,
  input logic [15:0]                     xs_wdata_i,
  input logic                            cmd_valid_i,
  input logic                            cmd_wfmt_i,
  input logic [$clog2(MaxK/64+1)-1:0]    cmd_ngroups_i,
  input logic [RowBlkW-1:0]              cmd_nrowblk_i,
  input logic                            w_valid_i,
  input logic [Lanes*4-1:0]              w_data_i,
  input logic                            ws_valid_i,
  input logic [15:0]                     ws_data_i,
  input logic                            y_ready_i,
  input logic                            status_clr_i
);

  logic        cmd_ready_o, w_ready_o, ws_ready_o, y_valid_o, busy_o;
  logic [31:0] y_data_o;

  bpu_qmv_slice #(
    .Lanes(Lanes), .RowInterleave(RowInterleave), .MaxK(MaxK), .RowBlkW(RowBlkW),
    .OutFifoDepth(OutFifoDepth), .ProdReg(1'b0), .TreeRegEvery(0), .I2fReg(1'b0),
    .MulPipe(3'b001), .AddPipe(3'b001), .EnPerf(1'b0)
  ) dut (
    .clk_i, .rst_ni,
    .x_we_i, .x_waddr_i, .x_wdata_i, .xs_we_i, .xs_waddr_i, .xs_wdata_i,
    .cmd_valid_i, .cmd_ready_o, .cmd_wfmt_i, .cmd_ngroups_i, .cmd_nrowblk_i,
    .w_valid_i, .w_ready_o, .w_data_i,
    .ws_valid_i, .ws_ready_o, .ws_data_i,
    .y_valid_o, .y_ready_i, .y_data_o,
    .busy_o,
    .status_clr_i, .err_cmd_o(), .flag_nan_o(), .flag_inf_o(),
    .perf_beats_o(), .perf_stall_w_o(), .perf_stall_ws_o(), .perf_stall_out_o()
  );

  // Previous-cycle copies (explicit registers instead of $past for frontend portability).
  logic        init_q = 1'b1;
  logic        rst_n_q, cmd_valid_q, cmd_ready_q, w_valid_q, w_ready_q, ws_valid_q, ws_ready_q;
  logic        y_valid_q, y_ready_q;
  logic [31:0] y_data_q;
  logic [Lanes*4-1:0] w_data_q;
  logic [15:0] ws_data_q;
  logic [$clog2(MaxK/64+1)+RowBlkW:0] cmd_q, cmd_now;

  assign cmd_now = {cmd_wfmt_i, cmd_ngroups_i, cmd_nrowblk_i};

  always_ff @(posedge clk_i) begin
    init_q      <= 1'b0;
    rst_n_q     <= rst_ni;
    cmd_valid_q <= cmd_valid_i;
    cmd_ready_q <= cmd_ready_o;
    cmd_q       <= cmd_now;
    w_valid_q   <= w_valid_i;
    w_ready_q   <= w_ready_o;
    w_data_q    <= w_data_i;
    ws_valid_q  <= ws_valid_i;
    ws_ready_q  <= ws_ready_o;
    ws_data_q   <= ws_data_i;
    y_valid_q   <= y_valid_o;
    y_ready_q   <= y_ready_i;
    y_data_q    <= y_data_o;
  end

  logic steady;   // out of reset for at least one full cycle
  assign steady = !init_q && rst_n_q && rst_ni;

  always_comb begin
    // Start in reset; once released, reset stays released.
    if (init_q) assume (!rst_ni);
    if (!init_q && rst_n_q) assume (rst_ni);

    if (steady) begin
      // Sources follow valid/ready rules: hold valid and data until accepted.
      if (cmd_valid_q && !cmd_ready_q) assume (cmd_valid_i && cmd_now == cmd_q);
      if (w_valid_q && !w_ready_q)     assume (w_valid_i && w_data_i == w_data_q);
      if (ws_valid_q && !ws_ready_q)   assume (ws_valid_i && ws_data_i == ws_data_q);

      // The output side keeps the same promise.
      if (y_valid_q && !y_ready_q) assert (y_valid_o && y_data_o == y_data_q);
    end

    // Activation buffers are written only while no operation streams.
    if (x_we_i || xs_we_i) assume (cmd_ready_o);
  end

  // Reachability: results flow, and a second command gets accepted.
  always_comb begin
    if (steady) begin
      cover (y_valid_o && y_ready_i);
      cover (cmd_valid_q && cmd_ready_q && !busy_o && cmd_valid_i);
    end
  end

endmodule
