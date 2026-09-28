// Shared constants and helpers for the BPU compute engines.
//
// Everything under "Numerics contract" is part of compute/docs/numerics.md and
// must be identical in every hardware configuration. Throughput knobs (lanes,
// interleave, pipeline registers) are module parameters, never package constants.
package bpu_compute_pkg;

  // Not every importer uses every constant.
  /* verilator lint_off UNUSEDPARAM */

  // ---------------------------------------------------------------------------
  // Numerics contract
  // ---------------------------------------------------------------------------

  // Weights per quantization group along the reduction (K) dimension.
  localparam int unsigned QmvGroup = 64;

  // Exact signed width of one group sum: |sum| <= 64 * 128 * 128 = 2^20.
  localparam int unsigned QmvIsumW = 22;

  // Weight formats (command field `wfmt`).
  localparam logic WfmtW4 = 1'b0;  // signed int4, two codes per byte, low nibble first
  localparam logic WfmtW8 = 1'b1;  // signed int8

  // Canonical quiet NaN produced by every fp32 unit.
  localparam logic [31:0] Fp32QNaN = 32'h7fc0_0000;

  // FVU opcodes (bpuref.fvu).
  localparam logic [4:0] FvuVadd = 5'd0,  FvuVsub = 5'd1,   FvuVmul = 5'd2,  FvuVmuls = 5'd3,
                         FvuVadds = 5'd4, FvuVaxpy = 5'd5,  FvuVmuladd = 5'd6, FvuVmulg = 5'd7,
                         FvuVsfu = 5'd8,  FvuVrbf16 = 5'd9, FvuVcopy = 5'd10, FvuVperm = 5'd11,
                         FvuVqclamp = 5'd12, FvuVsel = 5'd13,
                         FvuRsum = 5'd16, FvuRdot = 5'd17,  FvuRmax = 5'd18, FvuRamax = 5'd19,
                         FvuVvecmat = 5'd24;

  // Special-function unit function codes (bpuref.sfu: RCP, RSQRT, EXP2, EXP, LOG2).
  localparam logic [2:0] SfuRcp   = 3'd0;
  localparam logic [2:0] SfuRsqrt = 3'd1;
  localparam logic [2:0] SfuExp2  = 3'd2;
  localparam logic [2:0] SfuExp   = 3'd3;
  localparam logic [2:0] SfuLog2  = 3'd4;

  /* verilator lint_on UNUSEDPARAM */

  // Argmax ordering (bpuref.qmv.f32_order_key): an unsigned key where
  // -inf < ... < -0 < +0 < ... < +inf, and every NaN ranks below -inf.
  function automatic logic [31:0] f32_order_key(input logic [31:0] b);
    if (b[30:23] == 8'hff && b[22:0] != '0) f32_order_key = 32'd0;
    else if (b[31])                          f32_order_key = ~b;
    else                                     f32_order_key = b | 32'h8000_0000;
  endfunction

  // Inverse of f32_order_key; key 0 (NaN, or nothing selected) gives the canonical NaN.
  function automatic logic [31:0] f32_from_order_key(input logic [31:0] k);
    if (k == '0)       f32_from_order_key = Fp32QNaN;
    else if (k[31])    f32_from_order_key = k & 32'h7fff_ffff;
    else               f32_from_order_key = ~k;
  endfunction

  // fp32 holding an exact integer in [-127, 127] (a VQCLAMP result) -> int8.
  // Anything below 1 in magnitude reads as 0; out-of-range values saturate.
  function automatic logic [7:0] f32_to_i8(input logic [31:0] b);
    logic [23:0] sig;
    logic [7:0]  mag;
    sig = {1'b1, b[22:0]};
    if (b[30:23] < 8'd127)       mag = 8'd0;
    else if (b[30:23] > 8'd133)  mag = 8'd127;
    else                         mag = 8'(sig >> (8'd150 - b[30:23]));
    f32_to_i8 = b[31] ? 8'(-mag) : mag;
  endfunction

  // ---------------------------------------------------------------------------
  // Latency helpers, so parents can size their delay lines
  // ---------------------------------------------------------------------------

  // Latency of a unit whose three stage boundaries are optionally registered.
  function automatic int unsigned pipe3_latency(input logic [2:0] mask);
    pipe3_latency = int'(mask[0]) + int'(mask[1]) + int'(mask[2]);
  endfunction

  // Registered levels in bpu_add_tree with `n` inputs.
  function automatic int unsigned add_tree_latency(input int unsigned n,
                                                   input int unsigned reg_every);
    add_tree_latency = (reg_every == 0) ? 0 : $clog2(n) / reg_every;
  endfunction

endpackage
