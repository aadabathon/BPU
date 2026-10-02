// Shared constants and helpers for the BPU compute engines: the numerics contract.
//
// Everything under "Numerics contract" is part of compute/docs/numerics.md and
// must be identical in every hardware configuration. Encodings that are interface
// choices (opcodes, function codes, descriptors) live in bpu_isa_pkg. Throughput knobs (lanes,
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

  // Opcodes, function codes and descriptor layouts: bpu_isa_pkg (provisional).

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
