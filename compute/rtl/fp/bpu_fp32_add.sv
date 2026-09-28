// IEEE-754 binary32 adder.
//   Round to nearest even; subnormal inputs and outputs; canonical quiet NaN;
//   no exception flags. Bit-exact with numpy float32 (see docs/numerics.md).
//
// Three combinational stages; PipeMask[i] registers the boundary after stage i:
//   stage 0: classify, order by magnitude, align the smaller operand (G/R/S bits)
//   stage 1: add or subtract significands, count leading zeros
//   stage 2: normalize, round, pack, select special results
// Latency = popcount(PipeMask). Valid-only pipeline: it never stalls.
module bpu_fp32_add #(
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
  // Stage 0: classify, swap so |big| >= |small|, align
  // ---------------------------------------------------------------------------
  logic        a_sign, b_sign;
  logic [7:0]  a_exp, b_exp;
  logic [22:0] a_man, b_man;
  logic        a_inf, b_inf, a_nan, b_nan;

  assign {a_sign, a_exp, a_man} = a_i;
  assign {b_sign, b_exp, b_man} = b_i;

  assign a_inf = (a_exp == 8'hff) && (a_man == '0);
  assign b_inf = (b_exp == 8'hff) && (b_man == '0);
  assign a_nan = (a_exp == 8'hff) && (a_man != '0);
  assign b_nan = (b_exp == 8'hff) && (b_man != '0);

  logic        s0_nan, s0_inf, s0_inf_sign, s0_swap, s0_eff_sub;
  logic        big_sign, sml_sign;
  logic [7:0]  big_exp, sml_exp, big_eexp, sml_eexp, exp_diff;
  logic [22:0] big_man, sml_man;
  logic [26:0] big_ext, sml_ext, sml_al;
  logic [4:0]  align_sh;
  logic [53:0] align_wide;

  assign s0_nan      = a_nan | b_nan | (a_inf & b_inf & (a_sign ^ b_sign));
  assign s0_inf      = (a_inf | b_inf) & ~s0_nan;
  assign s0_inf_sign = a_inf ? a_sign : b_sign;

  // Magnitude order is the unsigned order of the low 31 encoding bits.
  assign s0_swap = (b_i[30:0] > a_i[30:0]);
  assign {big_sign, big_exp, big_man} = s0_swap ? b_i : a_i;
  assign {sml_sign, sml_exp, sml_man} = s0_swap ? a_i : b_i;

  // Subnormals behave as exponent 1 with a zero hidden bit.
  assign big_eexp = (big_exp == 8'd0) ? 8'd1 : big_exp;
  assign sml_eexp = (sml_exp == 8'd0) ? 8'd1 : sml_exp;
  assign exp_diff = big_eexp - sml_eexp;

  // Significands with hidden bit on bit 26 and guard/round/sticky below bit 3.
  assign big_ext = {big_exp != 8'd0, big_man, 3'b000};
  assign sml_ext = {sml_exp != 8'd0, sml_man, 3'b000};

  // Shifts of 27 or more leave only sticky bits.
  assign align_sh   = (exp_diff > 8'd27) ? 5'd27 : exp_diff[4:0];
  assign align_wide = {sml_ext, 27'd0} >> align_sh;
  assign sml_al     = align_wide[53:27] | {26'd0, |align_wide[26:0]};

  assign s0_eff_sub = big_sign ^ sml_sign;

  localparam int unsigned S0W = 5 + 8 + 27 + 27;

  logic           s1_valid;
  logic [S0W-1:0] s1_bus;

  bpu_delay #(.Width(1), .Depth(int'(PipeMask[0])), .Reset(1'b1)) u_v0 (
    .clk_i, .rst_ni, .d_i(valid_i), .q_o(s1_valid)
  );
  bpu_delay #(.Width(S0W), .Depth(int'(PipeMask[0]))) u_d0 (
    .clk_i, .rst_ni,
    .d_i({s0_nan, s0_inf, s0_inf_sign, big_sign, s0_eff_sub, big_eexp, big_ext, sml_al}),
    .q_o(s1_bus)
  );

  // ---------------------------------------------------------------------------
  // Stage 1: significand add/subtract (|big| >= |small|, so never negative)
  // ---------------------------------------------------------------------------
  logic        s1_nan, s1_inf, s1_inf_sign, s1_sign, s1_eff_sub, s1_zero_sign;
  logic [7:0]  s1_eexp;
  logic [26:0] s1_big, s1_sml;
  logic [27:0] s1_sum;
  logic [4:0]  s1_lz;

  assign {s1_nan, s1_inf, s1_inf_sign, s1_sign, s1_eff_sub, s1_eexp, s1_big, s1_sml} = s1_bus;

  assign s1_sum = s1_eff_sub ? ({1'b0, s1_big} - {1'b0, s1_sml})
                             : ({1'b0, s1_big} + {1'b0, s1_sml});

  bpu_lzc #(.Width(27)) u_lzc (.in_i(s1_sum[26:0]), .cnt_o(s1_lz));

  // Under RNE an exact zero is +0 unless both operands were -0.
  assign s1_zero_sign = s1_eff_sub ? 1'b0 : s1_sign;

  localparam int unsigned S1W = 5 + 8 + 28 + 5;

  logic           s2_valid;
  logic [S1W-1:0] s2_bus;

  bpu_delay #(.Width(1), .Depth(int'(PipeMask[1])), .Reset(1'b1)) u_v1 (
    .clk_i, .rst_ni, .d_i(s1_valid), .q_o(s2_valid)
  );
  bpu_delay #(.Width(S1W), .Depth(int'(PipeMask[1]))) u_d1 (
    .clk_i, .rst_ni,
    .d_i({s1_nan, s1_inf, s1_inf_sign, s1_sign, s1_zero_sign, s1_eexp, s1_sum, s1_lz}),
    .q_o(s2_bus)
  );

  // ---------------------------------------------------------------------------
  // Stage 2: normalize, round to nearest even, pack, specials
  // ---------------------------------------------------------------------------
  logic        s2_nan, s2_inf, s2_inf_sign, s2_sign, s2_zero_sign;
  logic [7:0]  s2_eexp, s2_lim;
  logic [27:0] s2_sum;
  logic [4:0]  s2_lz, s2_sh;
  logic [26:0] s2_norm;
  logic [8:0]  s2_exp;
  logic        s2_zero, s2_ovf, s2_round_up;
  logic [7:0]  s2_efield;
  logic [30:0] s2_mag;
  logic [31:0] s2_y;

  assign {s2_nan, s2_inf, s2_inf_sign, s2_sign, s2_zero_sign, s2_eexp, s2_sum, s2_lz} = s2_bus;

  assign s2_zero = (s2_sum == '0);
  // Left shifts stop at exponent 1; anything still unnormalized is subnormal.
  assign s2_lim  = s2_eexp - 8'd1;
  assign s2_sh   = ({3'b000, s2_lz} > s2_lim) ? s2_lim[4:0] : s2_lz;

  always_comb begin
    if (s2_sum[27]) begin
      // Carry out: shift right one, folding the dropped bit into sticky.
      s2_norm = {s2_sum[27:2], s2_sum[1] | s2_sum[0]};
      s2_exp  = {1'b0, s2_eexp} + 9'd1;
    end else begin
      s2_norm = s2_sum[26:0] << s2_sh;
      s2_exp  = {1'b0, s2_eexp} - {4'b0000, s2_sh};
    end
  end

  assign s2_ovf      = (s2_exp >= 9'd255);
  assign s2_efield   = s2_norm[26] ? s2_exp[7:0] : 8'd0;
  assign s2_round_up = s2_norm[2] & (s2_norm[1] | s2_norm[0] | s2_norm[3]);
  // A rounding carry out of the mantissa increments the exponent field.
  assign s2_mag      = {s2_efield, s2_norm[25:3]} + 31'(s2_round_up);

  always_comb begin
    if (s2_nan)       s2_y = Fp32QNaN;
    else if (s2_inf)  s2_y = {s2_inf_sign, 8'hff, 23'd0};
    else if (s2_ovf)  s2_y = {s2_sign, 8'hff, 23'd0};
    else if (s2_zero) s2_y = {s2_zero_sign, 31'd0};
    else              s2_y = {s2_sign, s2_mag};
  end

  bpu_delay #(.Width(1), .Depth(int'(PipeMask[2])), .Reset(1'b1)) u_v2 (
    .clk_i, .rst_ni, .d_i(s2_valid), .q_o(valid_o)
  );
  bpu_delay #(.Width(32), .Depth(int'(PipeMask[2]))) u_d2 (
    .clk_i, .rst_ni, .d_i(s2_y), .q_o(y_o)
  );

endmodule
