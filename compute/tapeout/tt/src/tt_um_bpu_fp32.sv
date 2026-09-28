// Tiny Tapeout learning run for the BPU compute engines: the IEEE fp32 adder and
// multiplier that every BPU datapath is built from (bpu_fp32_add / bpu_fp32_mul,
// unchanged, combinational), behind a byte-wide host protocol. Silicon results
// can be checked bit for bit against bpuref (numpy float32 + canonical NaN).
//
// Protocol (the demo board's microcontroller is the host). Every command acts on
// a rising edge of STROBE = uio_in[0], synchronized to clk:
//   uio_in[3:1]  CMD    0: A.byte[IDX] <= ui_in      1: B.byte[IDX] <= ui_in
//                       2: R <= A + B                3: R <= A - B
//                       4: R <= A * B                5: R <= A (I/O loopback)
//   uio_in[5:4]  IDX    byte index (0 = least significant)
//   uo_out              R.byte[IDX], continuously (IDX synchronized)
// A and B are stable for many cycles before an arithmetic command, and R is
// sampled once, so the combinational units have a whole clock period.
module tt_um_bpu_fp32 (
  input  wire [7:0] ui_in,
  output wire [7:0] uo_out,
  input  wire [7:0] uio_in,
  output wire [7:0] uio_out,
  output wire [7:0] uio_oe,
  input  wire       ena,
  input  wire       clk,
  input  wire       rst_n
);

  // Two-flop synchronizers for the host-driven controls; data (ui_in) is stable
  // around the strobe by protocol.
  logic [1:0] strobe_sync_q;
  logic       strobe_prev_q;
  logic [4:0] ctl_sync0_q, ctl_sync1_q;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      strobe_sync_q <= '0;
      strobe_prev_q <= 1'b0;
      ctl_sync0_q   <= '0;
      ctl_sync1_q   <= '0;
    end else begin
      strobe_sync_q <= {strobe_sync_q[0], uio_in[0]};
      strobe_prev_q <= strobe_sync_q[1];
      ctl_sync0_q   <= uio_in[5:1];
      ctl_sync1_q   <= ctl_sync0_q;
    end
  end

  logic       fire;
  logic [2:0] cmd;
  logic [1:0] idx;
  assign fire = strobe_sync_q[1] && !strobe_prev_q;
  assign cmd  = ctl_sync1_q[2:0];
  assign idx  = ctl_sync1_q[4:3];

  logic [31:0] a_q, b_q, r_q, sum, prod;

  bpu_fp32_add #(.PipeMask(3'b000)) u_add (
    .clk_i(clk), .rst_ni(rst_n), .valid_i(1'b1),
    .a_i(a_q), .b_i((cmd == 3'd3) ? {~b_q[31], b_q[30:0]} : b_q), .valid_o(), .y_o(sum));
  bpu_fp32_mul #(.PipeMask(3'b000)) u_mul (
    .clk_i(clk), .rst_ni(rst_n), .valid_i(1'b1), .a_i(a_q), .b_i(b_q), .valid_o(), .y_o(prod));

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_q <= '0;
      b_q <= '0;
      r_q <= '0;
    end else if (fire && ena) begin
      unique case (cmd)
        3'd0:        a_q[idx*8 +: 8] <= ui_in;
        3'd1:        b_q[idx*8 +: 8] <= ui_in;
        3'd2, 3'd3:  r_q <= sum;
        3'd4:        r_q <= prod;
        3'd5:        r_q <= a_q;
        default:     ;
      endcase
    end
  end

  assign uo_out  = r_q[idx*8 +: 8];
  assign uio_out = 8'h00;
  assign uio_oe  = 8'h00;           // all bidirectional pins are inputs

  logic unused;
  assign unused = &{1'b0, uio_in[7:6]};

endmodule
