// IEEE-754 binary32 multiplier.
//   Round to nearest even; subnormal inputs and outputs; canonical quiet NaN;
//   no exception flags. Bit-exact with numpy float32 (see docs/numerics.md).
//
// Three combinational stages; PipeMask[i] registers the boundary after stage i:
//   stage 0: unpack, classify, 24x24 significand product
//   stage 1: normalize, align subnormal results, collect guard/sticky
//   stage 2: round, pack, select special results
// Latency = popcount(PipeMask). Valid-only pipeline: it never stalls.
module bpu_fp32_mul #(
  parameter logic [2:0] PipeMask = 3'b111
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        valid_i,
  input  logic [31:0] a_i,
  input  logic [31:0] b_i,
  output logic        valid_o,
  output logic [31:0] y_o
);

  import bpu_compute_pkg::*;

  // ---------------------------------------------------------------------------
  // Stage 0: unpack, classify, significand product
  // ---------------------------------------------------------------------------
  logic        a_sign, b_sign;
  logic [7:0]  a_exp, b_exp;
  logic [22:0] a_man, b_man;
  logic        a_zero, b_zero, a_inf, b_inf, a_nan, b_nan;

  assign {a_sign, a_exp, a_man} = a_i;
  assign {b_sign, b_exp, b_man} = b_i;

  assign a_zero = (a_exp == 8'd0)   && (a_man == '0);
  assign b_zero = (b_exp == 8'd0)   && (b_man == '0);
  assign a_inf  = (a_exp == 8'hff)  && (a_man == '0);
  assign b_inf  = (b_exp == 8'hff)  && (b_man == '0);
  assign a_nan  = (a_exp == 8'hff)  && (a_man != '0);
  assign b_nan  = (b_exp == 8'hff)  && (b_man != '0);

  logic               s0_sign, s0_nan, s0_inf, s0_zero;
  logic signed [10:0] s0_exp;   // biased exponent if the product MSB lands on bit 47
  logic [47:0]        s0_prod;

  assign s0_sign = a_sign ^ b_sign;
  assign s0_nan  = a_nan | b_nan | (a_inf & b_zero) | (a_zero & b_inf);
  assign s0_inf  = (a_inf | b_inf) & ~s0_nan;
  assign s0_zero = (a_zero | b_zero) & ~s0_nan & ~s0_inf;

  // Subnormals behave as exponent 1 with a zero hidden bit.
  // value = sig_a * sig_b * 2^(ea + eb - 254 - 46); with the MSB on bit 47 the
  // biased exponent is ea + eb - 126.
  assign s0_exp  = $signed({3'b000, (a_exp == 8'd0) ? 8'd1 : a_exp})
                 + $signed({3'b000, (b_exp == 8'd0) ? 8'd1 : b_exp})
                 - 11'sd126;
  assign s0_prod = {a_exp != 8'd0, a_man} * {b_exp != 8'd0, b_man};

  localparam int unsigned S0W = 4 + 11 + 48;

  logic           s1_valid;
  logic [S0W-1:0] s1_bus;

  bpu_delay #(.Width(1), .Depth(int'(PipeMask[0])), .Reset(1'b1)) u_v0 (
    .clk_i, .rst_ni, .d_i(valid_i), .q_o(s1_valid)
  );
  bpu_delay #(.Width(S0W), .Depth(int'(PipeMask[0]))) u_d0 (
    .clk_i, .rst_ni,
    .d_i({s0_sign, s0_nan, s0_inf, s0_zero, s0_exp, s0_prod}),
    .q_o(s1_bus)
  );

  // ---------------------------------------------------------------------------
  // Stage 1: normalize, subnormal alignment
  // ---------------------------------------------------------------------------
  logic               s1_sign, s1_nan, s1_inf, s1_zero;
  logic signed [10:0] s1_exp_in, s1_exp;
  logic [47:0]        s1_prod, s1_norm;
  logic [5:0]         s1_lz, s1_dsh;
  logic [95:0]        s1_wide;
  logic               s1_ovf, s1_guard, s1_sticky;
  logic [7:0]         s1_efield;
  logic [22:0]        s1_keep;
  logic               unused_hidden;

  assign {s1_sign, s1_nan, s1_inf, s1_zero, s1_exp_in, s1_prod} = s1_bus;

  bpu_lzc #(.Width(48)) u_lzc (.in_i(s1_prod), .cnt_o(s1_lz));

  assign s1_norm = s1_prod << s1_lz;
  assign s1_exp  = s1_exp_in - $signed({5'b0, s1_lz});
  assign s1_ovf  = (s1_exp > 11'sd254);

  // Results below the normal range shift right until the exponent reaches 1
  // (encoded as field 0). Shifts of 48 or more leave only sticky bits.
  always_comb begin
    if (s1_exp >= 11'sd1)        s1_dsh = 6'd0;
    else if (s1_exp <= -11'sd47) s1_dsh = 6'd48;
    else                         s1_dsh = 6'(11'sd1 - s1_exp);
  end

  assign s1_wide   = {s1_norm, 48'd0} >> s1_dsh;
  // The hidden bit (s1_wide[95]) is implied by the exponent field.
  assign unused_hidden = s1_wide[95];
  assign s1_keep   = s1_wide[94:72];
  assign s1_guard  = s1_wide[71];
  assign s1_sticky = |s1_wide[70:0];
  assign s1_efield = (s1_exp >= 11'sd1) ? s1_exp[7:0] : 8'd0;

  localparam int unsigned S1W = 5 + 8 + 23 + 2;

  logic           s2_valid;
  logic [S1W-1:0] s2_bus;

  bpu_delay #(.Width(1), .Depth(int'(PipeMask[1])), .Reset(1'b1)) u_v1 (
    .clk_i, .rst_ni, .d_i(s1_valid), .q_o(s2_valid)
  );
  bpu_delay #(.Width(S1W), .Depth(int'(PipeMask[1]))) u_d1 (
    .clk_i, .rst_ni,
    .d_i({s1_sign, s1_nan, s1_inf, s1_zero, s1_ovf, s1_efield, s1_keep, s1_guard, s1_sticky}),
    .q_o(s2_bus)
  );

  // ---------------------------------------------------------------------------
  // Stage 2: round to nearest even, pack, specials
  // ---------------------------------------------------------------------------
  logic        s2_sign, s2_nan, s2_inf, s2_zero, s2_ovf, s2_guard, s2_sticky, s2_round_up;
  logic [7:0]  s2_efield;
  logic [22:0] s2_man;
  logic [30:0] s2_mag;
  logic [31:0] s2_y;

  assign {s2_sign, s2_nan, s2_inf, s2_zero, s2_ovf, s2_efield, s2_man, s2_guard, s2_sticky} = s2_bus;

  assign s2_round_up = s2_guard & (s2_sticky | s2_man[0]);
  // A rounding carry out of the mantissa increments the exponent field. That covers
  // mantissa overflow, subnormal -> normal, and max-normal -> infinity.
  assign s2_mag = {s2_efield, s2_man} + 31'(s2_round_up);

  always_comb begin
    if (s2_nan)                s2_y = Fp32QNaN;
    else if (s2_inf || s2_ovf) s2_y = {s2_sign, 8'hff, 23'd0};
    else if (s2_zero)          s2_y = {s2_sign, 31'd0};
    else                       s2_y = {s2_sign, s2_mag};
  end

  bpu_delay #(.Width(1), .Depth(int'(PipeMask[2])), .Reset(1'b1)) u_v2 (
    .clk_i, .rst_ni, .d_i(s2_valid), .q_o(valid_o)
  );
  bpu_delay #(.Width(32), .Depth(int'(PipeMask[2]))) u_d2 (
    .clk_i, .rst_ni, .d_i(s2_y), .q_o(y_o)
  );

endmodule
