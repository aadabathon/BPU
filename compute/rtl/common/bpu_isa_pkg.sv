// Provisional encodings: opcodes, function codes and the command descriptor layout.
//
// These are interface choices, not numerics (compare bpu_compute_pkg), and are
// expected to change when the ISA / control teams settle the command format. The
// Python mirror is bpuref/isa.py; change both together.
package bpu_isa_pkg;

  /* verilator lint_off UNUSEDPARAM */

  // ---------------------------------------------------------------------------
  // Engine operations
  // ---------------------------------------------------------------------------

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

  // ---------------------------------------------------------------------------
  // Command descriptors (control -> bpu_cmd_seq)
  // ---------------------------------------------------------------------------
  // A descriptor is {unit, tag, wait mask, body}. Addresses and strides are 32-bit
  // element (fp32) addresses into the shared SRAM.

  localparam logic [1:0] UnitVec = 2'd0;      // vector unit (bpu_fvu)
  localparam logic [1:0] UnitMat = 2'd1;      // matrix unit (bpu_qmv_engine)
  localparam logic [1:0] UnitMem = 2'd2;      // memory manager (outside the core)

  localparam int unsigned BodyW = 432;

  // Vector body, LSB first: op, func, half_log2, rows, cols, then 32-bit
  // d a b c s t addresses and their row strides ds as bs cs ss ts.
  localparam int unsigned VbOp = 0, VbFunc = 5, VbHalf = 8, VbRows = 13, VbCols = 29;
  localparam int unsigned VbAddr = 45;        // field i (d a b c s t ds .. ts) at VbAddr + 32 i

  // Matrix body: wid, wfmt, argmax, k, n, then 32-bit x, xs, y.
  localparam int unsigned MbWid = 0, MbWfmt = 16, MbArgmax = 17, MbK = 18, MbN = 34;
  localparam int unsigned MbNW = 24;
  localparam int unsigned MbX = 58, MbXs = 90, MbY = 122;

  // Memory body: opaque to the core, defined by the memory manager.

  /* verilator lint_on UNUSEDPARAM */

endpackage
