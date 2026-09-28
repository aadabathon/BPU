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
