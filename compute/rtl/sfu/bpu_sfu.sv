// Special-function unit: rcp, rsqrt, exp2, exp, log2 on fp32, one per cycle.
// Bit-exact with bpuref.sfu.sfu (docs/numerics.md, "SFU"): range reduction,
// a 128-segment quadratic from bpu_sfu_rom, and fixed-point packing.
// Subnormal inputs read as zero; results below the normal range flush to zero.
//
// Five combinational stages; PipeMask[i] registers the boundary after stage i:
//   0: unpack, specials, fixed-point range reduction, table select
//   1: exp: x * log2(e); table lookup
//   2: quadratic  y = c0 + c1*d + c2*d^2
//   3: clamp / exact cases; log2: t * h + e'
//   4: round to nearest even, pack, special results
// Latency = popcount(PipeMask). Valid-only pipeline: it never stalls.
module bpu_sfu #(
  parameter logic [4:0] PipeMask = 5'b11111
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        valid_i,
  input  logic [2:0]  func_i,    // bpu_compute_pkg::Sfu*
  input  logic [31:0] a_i,
  output logic        valid_o,
  output logic [31:0] y_o
);

  import bpu_compute_pkg::*;
  import bpu_isa_pkg::*;
  import bpu_sfu_rom_pkg::*;

  localparam logic [2:0] TRcp = 3'd0, TRsqEven = 3'd1, TRsqOdd = 3'd2, TExp2 = 3'd3,
                         TLogPos = 3'd4, TLogNeg = 3'd5;
  localparam logic signed [31:0] Log2eQ = 32'sd1549082005;   // round(log2(e) * 2^30)

  // ---------------------------------------------------------------------------
  // Stage 0: unpack, specials, range reduction
  // ---------------------------------------------------------------------------
  logic        a_sign, a_nan, a_inf, a_zero, a_big, m_zero, lo, odd, exact;
  logic [7:0]  a_exp;
  logic [22:0] a_man;
  logic signed [9:0] sh, E, h, ep, expf0;
  logic [23:0] sig, u0, v_int;
  logic [31:0] mag;
  logic signed [32:0] X;
  logic signed [24:0] t0;
  logic [2:0]  tbl0;

  assign {a_sign, a_exp, a_man} = a_i;
  assign a_nan  = (a_exp == 8'hff) && (a_man != '0);
  assign a_inf  = (a_exp == 8'hff) && (a_man == '0);
  assign a_zero = (a_exp == 8'd0);                      // DAZ
  assign m_zero = (a_man == '0);

  // exp / exp2: x as fixed point with 24 fraction bits, truncated toward zero.
  // |x| >= 256 (a_big) always over- or underflows and is resolved in stage 4.
  assign a_big = (a_exp >= 8'd135);
  assign sh    = $signed({2'b00, a_exp}) - 10'sd126;
  assign sig   = a_zero ? 24'd0 : {1'b1, a_man};
  always_comb begin
    if (sh >= 0)        mag = 32'(sig) << sh[3:0];
    else if (sh < -31)  mag = '0;
    else                mag = 32'(sig) >> (-sh);
  end
  assign X = a_sign ? -$signed({1'b0, mag}) : $signed({1'b0, mag});

  // rcp / rsqrt: exponent of the result; exact powers of two bypass the table.
  assign E     = $signed({2'b00, a_exp}) - 10'sd127;
  assign odd   = E[0];
  assign h     = E >>> 1;
  assign exact = !odd && m_zero;

  // log2: 1.m < 1.5 -> t = m; else t = -(1 - m)/2 with e' = E + 1. ~m = 2^23 - 1 - m.
  assign lo    = !a_man[22];
  assign v_int = 24'h80_0000 - {1'b0, a_man};
  assign ep    = lo ? E : E + 10'sd1;
  assign t0    = lo ? $signed({1'b0, a_man, 1'b0}) : -$signed({1'b0, v_int});

  always_comb begin
    unique case (func_i)
      SfuRcp: begin
        tbl0  = TRcp;
        u0    = {a_man, 1'b0};
        expf0 = m_zero ? 10'sd254 - $signed({2'b00, a_exp}) : 10'sd253 - $signed({2'b00, a_exp});
      end
      SfuRsqrt: begin
        tbl0  = odd ? TRsqOdd : TRsqEven;
        u0    = {a_man, 1'b0};
        expf0 = exact ? 10'sd127 - h : 10'sd126 - h;
      end
      SfuLog2: begin
        tbl0  = lo ? TLogPos : TLogNeg;
        u0    = lo ? {a_man[21:0], 2'b00} : {~a_man[21:0], 2'b00};
        expf0 = ep;
      end
      default: begin            // exp2, exp: table argument known after stage 1
        tbl0  = TExp2;
        u0    = '0;
        expf0 = '0;
      end
    endcase
  end

  // Stage 0 -> 1 bus
  localparam int unsigned B1W = 3 + 7 + 33 + 3 + 24 + 10 + 25;
  logic           v1;
  logic [B1W-1:0] b1;
  logic [2:0]     f1, tbl1;
  logic           s1_sign, s1_nan, s1_inf, s1_zero, s1_big, s1_mzero, s1_exact;
  logic signed [32:0] X1;
  logic [23:0]    u1;
  logic signed [9:0]  expf1;
  logic signed [24:0] t1;

  bpu_delay #(.Width(1), .Depth(int'(PipeMask[0])), .Reset(1'b1)) u_v1 (
    .clk_i, .rst_ni, .d_i(valid_i), .q_o(v1));
  bpu_delay #(.Width(B1W), .Depth(int'(PipeMask[0]))) u_b1 (
    .clk_i, .rst_ni,
    .d_i({func_i, a_sign, a_nan, a_inf, a_zero, a_big, m_zero, exact, X, tbl0, u0, expf0, t0}),
    .q_o(b1));
  assign {f1, s1_sign, s1_nan, s1_inf, s1_zero, s1_big, s1_mzero, s1_exact, X1, tbl1, u1, expf1, t1} = b1;

  // ---------------------------------------------------------------------------
  // Stage 1: exp scaling, table lookup
  // ---------------------------------------------------------------------------
  logic signed [64:0] xl;
  logic signed [34:0] Y;
  logic signed [10:0] n1;
  logic [23:0]        u1x;
  logic [2:0]         tbl1x;
  logic signed [SfuC0W-1:0] c0;
  logic signed [SfuC1W-1:0] c1;
  logic signed [SfuC2W-1:0] c2;

  assign xl    = X1 * Log2eQ;
  assign Y     = (f1 == SfuExp) ? 35'(xl >>> 30) : 35'(X1);
  assign n1    = 11'(Y >>> 24);
  assign u1x   = (f1 == SfuExp || f1 == SfuExp2) ? Y[23:0] : u1;
  assign tbl1x = tbl1;

  bpu_sfu_rom u_rom (.table_i(tbl1x), .idx_i(u1x[23:17]), .c0_o(c0), .c1_o(c1), .c2_o(c2));

  // Stage 1 -> 2 bus
  localparam int unsigned B2W = 3 + 7 + 11 + 10 + 25 + SfuC0W + SfuC1W + SfuC2W + 17;
  logic           v2;
  logic [B2W-1:0] b2;
  logic [2:0]     f2;
  logic           s2_sign, s2_nan, s2_inf, s2_zero, s2_big, s2_mzero, s2_exact;
  logic signed [10:0] n2;
  logic signed [9:0]  expf2;
  logic signed [24:0] t2;
  logic signed [SfuC0W-1:0] c0_2;
  logic signed [SfuC1W-1:0] c1_2;
  logic signed [SfuC2W-1:0] c2_2;
  logic [16:0]    d2;

  bpu_delay #(.Width(1), .Depth(int'(PipeMask[1])), .Reset(1'b1)) u_v2 (
    .clk_i, .rst_ni, .d_i(v1), .q_o(v2));
  bpu_delay #(.Width(B2W), .Depth(int'(PipeMask[1]))) u_b2 (
    .clk_i, .rst_ni,
    .d_i({f1, s1_sign, s1_nan, s1_inf, s1_zero, s1_big, s1_mzero, s1_exact, n1, expf1, t1,
          c0, c1, c2, u1x[16:0]}),
    .q_o(b2));
  assign {f2, s2_sign, s2_nan, s2_inf, s2_zero, s2_big, s2_mzero, s2_exact, n2, expf2, t2,
          c0_2, c1_2, c2_2, d2} = b2;

  // ---------------------------------------------------------------------------
  // Stage 2: y = c0 + floor(c1*d / 2^17) + floor(c2*floor(d^2 / 2^17) / 2^17)
  // ---------------------------------------------------------------------------
  logic [33:0]        dd;
  logic [16:0]        sq;
  logic signed [SfuC1W+17:0] p1;
  logic signed [SfuC2W+17:0] p2;
  logic signed [32:0] y2;

  logic        unused_dd;
  assign dd = d2 * d2;
  assign sq = dd[33:17];
  assign unused_dd = ^dd[16:0];
  assign p1 = c1_2 * $signed({1'b0, d2});
  assign p2 = c2_2 * $signed({1'b0, sq});
  assign y2 = 33'(c0_2) + 33'(p1 >>> 17) + 33'(p2 >>> 17);

  // Stage 2 -> 3 bus
  localparam int unsigned B3W = 3 + 7 + 11 + 10 + 25 + 33;
  logic           v3;
  logic [B3W-1:0] b3;
  logic [2:0]     f3;
  logic           s3_sign, s3_nan, s3_inf, s3_zero, s3_big, s3_mzero, s3_exact;
  logic signed [10:0] n3;
  logic signed [9:0]  expf3;
  logic signed [24:0] t3;
  logic signed [32:0] y3;

  bpu_delay #(.Width(1), .Depth(int'(PipeMask[2])), .Reset(1'b1)) u_v3 (
    .clk_i, .rst_ni, .d_i(v2), .q_o(v3));
  bpu_delay #(.Width(B3W), .Depth(int'(PipeMask[2]))) u_b3 (
    .clk_i, .rst_ni,
    .d_i({f2, s2_sign, s2_nan, s2_inf, s2_zero, s2_big, s2_mzero, s2_exact, n2, expf2, t2, y2}),
    .q_o(b3));
  assign {f3, s3_sign, s3_nan, s3_inf, s3_zero, s3_big, s3_mzero, s3_exact, n3, expf3, t3, y3} = b3;

  // ---------------------------------------------------------------------------
  // Stage 3: clamp to [1, 2) and exact cases; log2 fixed-point result
  // ---------------------------------------------------------------------------
  logic [30:0]        yc;
  logic signed [9:0]  expf3x;
  logic signed [57:0] tp;
  logic signed [63:0] R;

  always_comb begin
    if ((f3 == SfuRcp && s3_mzero) || (f3 == SfuRsqrt && s3_exact)) yc = 31'h4000_0000;
    else if (y3 < 33'sh0_4000_0000)                                 yc = 31'h4000_0000;
    else if (y3 > 33'sh0_7fff_ffff)                                 yc = 31'h7fff_ffff;
    else                                                            yc = y3[30:0];
  end
  assign expf3x = (f3 == SfuExp || f3 == SfuExp2) ? 10'(n3 + 11'sd127) : expf3;
  assign tp     = t3 * y3;
  assign R      = (64'(expf3) <<< 54) + 64'(tp);            // log2: expf3 carries e'

  // Stage 3 -> 4 bus
  localparam int unsigned B4W = 3 + 5 + 11 + 10 + 31 + 64;
  logic           v4;
  logic [B4W-1:0] b4;
  logic [2:0]     f4;
  logic           s4_sign, s4_nan, s4_inf, s4_zero, s4_big;
  logic signed [10:0] n4;
  logic signed [9:0]  expf4;
  logic [30:0]    yc4;
  logic signed [63:0] R4;

  bpu_delay #(.Width(1), .Depth(int'(PipeMask[3])), .Reset(1'b1)) u_v4 (
    .clk_i, .rst_ni, .d_i(v3), .q_o(v4));
  bpu_delay #(.Width(B4W), .Depth(int'(PipeMask[3]))) u_b4 (
    .clk_i, .rst_ni,
    .d_i({f3, s3_sign, s3_nan, s3_inf, s3_zero, s3_big, n3, expf3x, yc, R}),
    .q_o(b4));
  assign {f4, s4_sign, s4_nan, s4_inf, s4_zero, s4_big, n4, expf4, yc4, R4} = b4;

  // ---------------------------------------------------------------------------
  // Stage 4: round to nearest even, pack, special results
  // ---------------------------------------------------------------------------
  // Unit-range result (rcp, rsqrt, exp2, exp): yc4 has its hidden bit on bit 30.
  logic        up_u;
  logic [30:0] mag_u;

  logic        unused_hidden;
  assign unused_hidden = yc4[30];                // implied by the exponent field
  assign up_u  = yc4[6] && ((yc4[5:0] != '0) || yc4[7]);
  assign mag_u = {expf4[7:0], yc4[29:7]} + 31'(up_u);

  // log2: signed fixed point with 54 fraction bits -> fp32.
  logic        r_sign, up_l;
  logic [63:0] r_mag, r_norm;
  logic [6:0]  r_lz;
  logic [7:0]  r_expf;
  logic [30:0] mag_l;
  logic        unused_l;

  assign r_sign = R4[63];
  assign r_mag  = r_sign ? 64'(-R4) : 64'(R4);
  bpu_lzc #(.Width(64)) u_lzc (.in_i(r_mag), .cnt_o(r_lz));
  assign r_norm = r_mag << r_lz[5:0];
  assign r_expf = 8'(8'd136 - {1'b0, r_lz});
  assign up_l   = r_norm[39] && ((r_norm[38:0] != '0) || r_norm[40]);
  assign mag_l  = {r_expf, r_norm[62:40]} + 31'(up_l);
  assign unused_l = r_norm[63];

  logic [31:0] y4;
  always_comb begin
    unique case (f4)
      SfuRcp: begin
        if (s4_nan)                 y4 = Fp32QNaN;
        else if (s4_inf)            y4 = {s4_sign, 31'd0};
        else if (s4_zero)           y4 = {s4_sign, 8'hff, 23'd0};
        else if (expf4 <= 10'sd0)   y4 = {s4_sign, 31'd0};                  // FTZ
        else                        y4 = {s4_sign, mag_u};
      end
      SfuRsqrt: begin
        if (s4_nan)                 y4 = Fp32QNaN;
        else if (s4_sign && !s4_zero) y4 = Fp32QNaN;                        // x < 0
        else if (s4_inf)            y4 = 32'd0;                             // +inf -> +0
        else if (s4_zero)           y4 = {s4_sign, 8'hff, 23'd0};           // +-0 -> +-inf
        else                        y4 = {1'b0, mag_u};
      end
      SfuLog2: begin
        if (s4_nan)                 y4 = Fp32QNaN;
        else if (s4_sign && !s4_zero) y4 = Fp32QNaN;
        else if (s4_inf)            y4 = 32'h7f80_0000;
        else if (s4_zero)           y4 = 32'hff80_0000;                     // log2(+-0) = -inf
        else if (R4 == '0)          y4 = 32'd0;
        else                        y4 = {r_sign, mag_l};
      end
      default: begin                                                        // exp2, exp
        if (s4_nan)                 y4 = Fp32QNaN;
        else if (s4_inf || s4_big)  y4 = s4_sign ? 32'd0 : 32'h7f80_0000;
        else if (n4 < -11'sd126)    y4 = 32'd0;                             // FTZ
        else if (n4 >= 11'sd128)    y4 = 32'h7f80_0000;
        else                        y4 = {1'b0, mag_u};
      end
    endcase
  end

  bpu_delay #(.Width(1), .Depth(int'(PipeMask[4])), .Reset(1'b1)) u_v5 (
    .clk_i, .rst_ni, .d_i(v4), .q_o(valid_o));
  bpu_delay #(.Width(32), .Depth(int'(PipeMask[4]))) u_y5 (
    .clk_i, .rst_ni, .d_i(y4), .q_o(y_o));

endmodule
