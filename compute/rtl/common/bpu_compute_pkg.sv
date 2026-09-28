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

  /* verilator lint_on UNUSEDPARAM */

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
