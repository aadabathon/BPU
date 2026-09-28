// One FVU lane: fp32 multiply -> add, special functions, and simple
// element ops. Every result appears exactly Latency cycles after its inputs
// (Latency = max(mul + add, sfu)), so the FVU can track items with one delay line.
//
// Lane ops (lop_i) reuse the FVU opcode values where they coincide:
//   ADD a+b   SUB a-b   MUL a*b   MULS a*s   ADDS a+s   AXPY (s*a)+b   MULADD (a*b)+c
//   MULG a*t  SFU f(a)  RBF16 bf16(a)  COPY a  QCLAMP clamp(rne(a),+-127)  SEL a>s?b:c
//   ABS |a|
// zero_b_i replaces b by +0 (VVECMAT's first row).
module bpu_fvu_lane #(
  parameter logic [2:0] MulPipe = 3'b111,
  parameter logic [2:0] AddPipe = 3'b111,
  parameter logic [4:0] SfuPipe = 5'b11111,
  parameter bit         HasSfu  = 1'b1        // 0: SFU results come from a shared bank (sf_ext_i)
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        valid_i,
  input  logic [4:0]  lop_i,
  input  logic [2:0]  func_i,
  input  logic        zero_b_i,
  input  logic [31:0] a_i,
  input  logic [31:0] b_i,
  input  logic [31:0] c_i,
  input  logic [31:0] s_i,
  input  logic [31:0] t_i,
  input  logic [31:0] sf_ext_i,             // shared-bank SFU result, Ls cycles after the inputs
  output logic [31:0] res_o
);

  import bpu_compute_pkg::*;

  localparam logic [4:0] LopAbs = 5'd14;
  localparam int unsigned Lm   = pipe3_latency(MulPipe);
  localparam int unsigned La   = pipe3_latency(AddPipe);
  localparam int unsigned Ls   = int'(SfuPipe[0]) + int'(SfuPipe[1]) + int'(SfuPipe[2])
                               + int'(SfuPipe[3]) + int'(SfuPipe[4]);
  localparam int unsigned Lma  = Lm + La;
  localparam int unsigned Ltot = (Lma > Ls) ? Lma : Ls;

  // ---------------------------------------------------------------------------
  // Multiply, then add
  // ---------------------------------------------------------------------------
  logic [31:0] my, m, m2;
  always_comb begin
    unique case (lop_i)
      FvuVmuls, FvuVaxpy: my = s_i;
      FvuVmulg:           my = t_i;
      default:            my = b_i;       // MUL, MULADD
    endcase
  end

  bpu_fp32_mul #(.PipeMask(MulPipe)) u_mul (
    .clk_i, .rst_ni, .valid_i, .a_i(a_i), .b_i(my), .valid_o(), .y_o(m)
  );

  logic [4:0]  lop_m;
  logic        zb_m;
  logic [31:0] a_m, b_m, c_m, s_m;
  bpu_delay #(.Width(5 + 1 + 128), .Depth(Lm)) u_dm (
    .clk_i, .rst_ni, .d_i({lop_i, zero_b_i, a_i, b_i, c_i, s_i}), .q_o({lop_m, zb_m, a_m, b_m, c_m, s_m})
  );

  logic [31:0] ax, ay, sum;
  always_comb begin
    unique case (lop_m)
      FvuVsub:    begin ax = a_m; ay = {~b_m[31], b_m[30:0]}; end
      FvuVadds:   begin ax = a_m; ay = s_m; end
      FvuVaxpy:   begin ax = m;   ay = zb_m ? 32'd0 : b_m; end
      FvuVmuladd: begin ax = m;   ay = c_m; end
      default:    begin ax = a_m; ay = b_m; end     // ADD
    endcase
  end

  bpu_fp32_add #(.PipeMask(AddPipe)) u_add (
    .clk_i, .rst_ni, .valid_i, .a_i(ax), .b_i(ay), .valid_o(), .y_o(sum)
  );
  bpu_delay #(.Width(32), .Depth(La)) u_dm2 (.clk_i, .rst_ni, .d_i(m), .q_o(m2));

  logic [4:0]  lop_ma;
  logic [31:0] arith, arith_t;
  bpu_delay #(.Width(5), .Depth(La)) u_dlop (.clk_i, .rst_ni, .d_i(lop_m), .q_o(lop_ma));
  always_comb begin
    unique case (lop_ma)
      FvuVmul, FvuVmuls, FvuVmulg: arith = m2;
      default:                     arith = sum;
    endcase
  end
  bpu_delay #(.Width(32), .Depth(Ltot - Lma)) u_da (.clk_i, .rst_ni, .d_i(arith), .q_o(arith_t));

  // ---------------------------------------------------------------------------
  // Special functions
  // ---------------------------------------------------------------------------
  logic [31:0] sf, sf_t;
  if (HasSfu) begin : g_sfu
    logic unused_ext;
    assign unused_ext = ^sf_ext_i;
    bpu_sfu #(.PipeMask(SfuPipe)) u_sfu (
      .clk_i, .rst_ni, .valid_i, .func_i, .a_i(a_i), .valid_o(), .y_o(sf)
    );
  end else begin : g_shared_sfu
    logic unused_func;
    assign unused_func = ^func_i;
    assign sf = sf_ext_i;
  end
  bpu_delay #(.Width(32), .Depth(Ltot - Ls)) u_ds (.clk_i, .rst_ni, .d_i(sf), .q_o(sf_t));

  // ---------------------------------------------------------------------------
  // Simple element ops (combinational, then delayed)
  // ---------------------------------------------------------------------------
  logic        a_nan, s_nan, gt;
  logic [32:0] rb;
  logic [31:0] rbf, qc, misc, misc_t;

  assign a_nan = (a_i[30:23] == 8'hff) && (a_i[22:0] != '0);
  assign s_nan = (s_i[30:23] == 8'hff) && (s_i[22:0] != '0);

  // bf16 round to nearest even; NaN -> canonical.
  assign rb  = {1'b0, a_i} + 33'h7fff + {32'd0, a_i[16]};
  assign rbf = a_nan ? Fp32QNaN : {rb[31:16], 16'h0000};
  logic unused_rb;
  assign unused_rb = ^{rb[32], rb[15:0]};

  // a > s under IEEE rules: false with NaN, and +0 == -0.
  assign gt = !a_nan && !s_nan && !((a_i[30:0] == '0) && (s_i[30:0] == '0))
           && (f32_order_key(a_i) > f32_order_key(s_i));

  // clamp(round-half-even(a), -127, 127) as fp32; NaN -> 0; zero is +0.
  logic [7:0]  q_mag;
  logic [8:0]  q_int;
  bpu_fvu_qround u_qround (.a_i(a_i[30:0]), .mag_o(q_mag));
  assign q_int = (a_i[31] && q_mag != '0) ? -{1'b0, q_mag} : {1'b0, q_mag};
  bpu_int2fp32 #(.InW(9), .Reg(1'b0)) u_q2f (
    .clk_i, .rst_ni, .valid_i, .x_i(q_int), .valid_o(), .y_o(qc)
  );

  always_comb begin
    unique case (lop_i)
      FvuVrbf16:   misc = rbf;
      FvuVqclamp:  misc = a_nan ? 32'd0 : qc;
      FvuVsel:     misc = gt ? b_i : c_i;
      LopAbs:      misc = {1'b0, a_i[30:0]};
      default:     misc = a_i;                     // COPY (and PERM, permuted upstream)
    endcase
  end
  bpu_delay #(.Width(32), .Depth(Ltot)) u_dmisc (.clk_i, .rst_ni, .d_i(misc), .q_o(misc_t));

  // ---------------------------------------------------------------------------
  // Result select
  // ---------------------------------------------------------------------------
  logic [4:0] lop_t;
  bpu_delay #(.Width(5), .Depth(Ltot)) u_dlop_t (.clk_i, .rst_ni, .d_i(lop_i), .q_o(lop_t));

  always_comb begin
    unique case (lop_t)
      FvuVadd, FvuVsub, FvuVmul, FvuVmuls, FvuVadds, FvuVaxpy, FvuVmuladd, FvuVmulg:
                 res_o = arith_t;
      FvuVsfu:   res_o = sf_t;
      default:   res_o = misc_t;
    endcase
  end

endmodule
