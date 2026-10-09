// Int8 systolic array: C = A * B with A stationary in the PEs and B streaming in
// from the top. Partial sums travel between PEs in carry-save form (separate sum
// and carry vectors) and are resolved by one carry-propagate adder per row, at
// the array's right edge.
//
// Modules (all in this file, no other dependencies):
//   bpu_sa_array   the array: A tile loading, B input skew, PE grid, row adders,
//                  output deskew
//   bpu_sa_pe      one processing element (port list from the 10/08 sketch)
//   bpu_sa_delay   a shift-register delay line
//
// ---------------------------------------------------------------------------------
// Geometry
//
//   A is a Rows x Cols tile; PE(i,k) holds A[i][k]. One B column (Cols int8 values)
//   produces one C column (Rows values):  C[i] = sum over k of A[i][k] * B[k].
//
//            B[0]     B[1]     B[2]          B[k] enters column k, delayed k cycles
//              |        |        |
//      0 -> PE(0,0) -> PE(0,1) -> PE(0,2) -> (+) -> C[0]    sum + carry move right
//              |        |        |
//      0 -> PE(1,0) -> PE(1,1) -> PE(1,2) -> (+) -> C[1]    B moves down
//
//   The array skews B on the way in and deskews C on the way out, so callers see
//   whole columns: one B column in per cycle, one C column out per cycle.
//
// Interface
//
//   A tile    a_valid_i / a_ready_o / a_row_i: one row of A per transfer, element k
//             in bits [8k+7:8k]. Send the LAST row first: rows shift down from the
//             top, so after Rows transfers the first row sent sits in row Rows-1.
//             The tile fills a shadow register in every PE, so the next tile can
//             load while the current one computes. a_full_o = a whole tile waits.
//   B         b_valid_i / b_col_i: one column of B per cycle, element k in bits
//             [8k+7:8k]. Always accepted (b_ready_o = 1). The first B column that
//             arrives while a_full_o is high switches to the new tile: that column
//             and every later one use it. The switch travels through the array with
//             that column, so earlier columns still in flight keep the old tile.
//             Before any tile has been loaded, outputs are undefined.
//   C         c_valid_o / c_col_o: C[i] (signed, AccW bits) in bits
//             [AccW*i +: AccW], Lat = Rows + Cols + 1 cycles after its B column was
//             accepted, in order. There is no backpressure: put a FIFO behind it if
//             the consumer can stall.
//   mode      mode_i = 0: int8 x int8. mode_i = 1 (int4) is reserved and not
//             implemented; err_mode_o is high while it is selected.
//
//   A tile becomes loadable again Rows + Cols - 2 cycles after the column that
//   switched to it, once that switch has reached every PE (a_ready_o handles it).
//
// Arithmetic
//
//   Signed int8 x signed int8, exact; sums are modulo 2^AccW, so AccW must be at
//   least 16 + clog2(Cols) for exact results (checked in simulation). The default
//   of 32 matches the "raw dot sums INT32" plan from the 10/03 compute meeting.
//
// Inside a PE (bpu_sa_pe): 8 partial products (sign-extended rows; the -2^7 row
// of B is negated as ~x + 1, with the +1 put in an empty bit of another row) plus
// the incoming sum and carry make 10 rows. Eight 3:2 carry-save adders, five
// levels deep, reduce them to 2 rows, which are registered. No carry propagates
// inside the array, so a PE's delay is a mux, an AND gate and five full adders,
// whatever AccW is.
//
// Knobs for the cost/benefit study:
//   * AccW: 32 is safe; 16 + clog2(Cols) is the minimum (fewer flops and adders).
//   * Sign extension: rows are fully sign-extended (simple, more full adders in the
//     high bits). The Baugh-Wooley or sign-extension-constant trick cuts that.
//   * Wallace (here) vs Dadda reduction: Dadda uses fewer full adders.
//   * Carry-save partial sums (here) double the psum flops and wires between PEs
//     but take the carry propagation out of every PE. The alternative resolves
//     sum + carry inside each PE (one adder per PE, half the psum flops).
//   * Valid-based clock gating: idle B slots are fed as zeros to cut toggling;
//     PE registers still clock every cycle.
// ---------------------------------------------------------------------------------

module bpu_sa_delay #(
  parameter int unsigned Width = 1,
  parameter int unsigned Depth = 1,
  parameter bit          Reset = 1'b0
) (
  input  logic             clk_i,
  input  logic             rst_ni,
  input  logic [Width-1:0] d_i,
  output logic [Width-1:0] q_o
);

  if (Depth == 0) begin : g_wire
    logic unused;
    assign unused = clk_i ^ rst_ni;
    assign q_o = d_i;
  end else begin : g_regs
    logic [Depth-1:0][Width-1:0] q;
    if (Reset) begin : g_rst
      always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
          q <= '0;
        end else begin
          q[0] <= d_i;
          for (int j = 1; j < Depth; j++) q[j] <= q[j-1];
        end
      end
    end else begin : g_norst
      logic unused;
      assign unused = rst_ni;
      always_ff @(posedge clk_i) begin
        q[0] <= d_i;
        for (int j = 1; j < Depth; j++) q[j] <= q[j-1];
      end
    end
    assign q_o = q[Depth-1];
  end

endmodule


// One processing element. Holds one element of A (active + shadow), passes B down,
// and adds A * B to the carry-save partial sum moving left to right.
module bpu_sa_pe #(
  parameter int unsigned AccW = 32
) (
  input  logic            clk_i,
  input  logic            rst_ni,
  input  logic            mode_i,      // 0: int8. 1: int4, reserved (see bpu_sa_array)

  // A: the shadow registers form a shift chain down each column
  input  logic            load_i,      // shift the chain (whole array at once)
  input  logic [7:0]      load_a_i,    // from the PE above, or the array's A input
  output logic [7:0]      load_a_o,    // to the PE below

  // B: moves down one PE per cycle; swap rides with it
  input  logic [7:0]      b_i,
  input  logic            swap_i,      // this B uses the shadow A, which becomes active
  output logic [7:0]      b_o,
  output logic            swap_o,

  // Partial sum in carry-save form: value = sum + carry (mod 2^AccW)
  input  logic [AccW-1:0] sum_i,
  input  logic [AccW-1:0] carry_i,
  output logic [AccW-1:0] sum_o,
  output logic [AccW-1:0] carry_o
);

  // 3:2 carry-save adder: one full adder per bit. x + y + z = sum + carry, with the
  // carry already shifted to its weight. Nothing ripples sideways, so the delay is
  // one full adder at any width. The carry out of the top bit falls off: modulo 2^AccW.
  function automatic logic [2*AccW-1:0] csa(input logic [AccW-1:0] x, y, z);
    logic [AccW-1:0] s, c;
    s = x ^ y ^ z;
    c = ((x & y) | (x & z) | (y & z)) << 1;
    return {c, s};
  endfunction

  // A registers
  logic [7:0] a_shadow_q, a_active_q, a_use;

  always_ff @(posedge clk_i) begin
    if (load_i) a_shadow_q <= load_a_i;
    if (swap_i) a_active_q <= a_shadow_q;
  end

  assign load_a_o = a_shadow_q;
  assign a_use    = swap_i ? a_shadow_q : a_active_q;

  // Partial products of a_use * b_i (signed x signed):
  //   b = -b7*2^7 + sum_{i<7} b_i*2^i, so
  //   a*b = sum_{i<7} b_i*(a << i) + b7*(~(a << 7) + 1)      (mod 2^AccW)
  // The +1 goes into bit 0 of row 1, which is always 0 there.
  logic [AccW-1:0]      a_ext;
  logic [7:0][AccW-1:0] pp;

  assign a_ext = {{(AccW-8){a_use[7]}}, a_use};

  always_comb begin
    for (int i = 0; i < 7; i++) pp[i] = b_i[i] ? (a_ext << i) : '0;
    pp[7]    = b_i[7] ? ~(a_ext << 7) : '0;
    pp[1][0] = b_i[7];
  end

  // Wallace reduction: 10 rows (8 partial products + sum_i + carry_i) -> 2 rows.
  //   level 1: 10 -> 7   level 2: 7 -> 5   level 3: 5 -> 4   level 4: 4 -> 3   level 5: 3 -> 2
  logic [AccW-1:0] s10, c10, s11, c11, s12, c12;
  logic [AccW-1:0] s20, c20, s21, c21;
  logic [AccW-1:0] s30, c30, s40, c40, s50, c50;

  assign {c10, s10} = csa(pp[0], pp[1], pp[2]);
  assign {c11, s11} = csa(pp[3], pp[4], pp[5]);
  assign {c12, s12} = csa(pp[6], pp[7], sum_i);

  assign {c20, s20} = csa(s10, c10, s11);
  assign {c21, s21} = csa(c11, s12, c12);

  assign {c30, s30} = csa(s20, c20, s21);
  assign {c40, s40} = csa(s30, c30, c21);
  assign {c50, s50} = csa(s40, c40, carry_i);

  always_ff @(posedge clk_i) begin
    b_o     <= b_i;
    sum_o   <= s50;
    carry_o <= c50;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) swap_o <= 1'b0;
    else         swap_o <= swap_i;
  end

  logic unused_mode;
  assign unused_mode = mode_i;

endmodule


module bpu_sa_array #(
  parameter int unsigned Rows = 8,     // rows of the A tile = outputs per B column
  parameter int unsigned Cols = 8,     // columns of the A tile = reduction length
  parameter int unsigned AccW = 32     // accumulator width, >= 16 + clog2(Cols)
) (
  input  logic                 clk_i,
  input  logic                 rst_ni,

  input  logic                 mode_i,      // 0: int8; 1: int4 (not implemented)
  output logic                 err_mode_o,

  // A tile, one row per transfer, last row first
  input  logic                 a_valid_i,
  output logic                 a_ready_o,
  input  logic [Cols*8-1:0]    a_row_i,
  output logic                 a_full_o,

  // B, one column per cycle
  input  logic                 b_valid_i,
  output logic                 b_ready_o,
  input  logic [Cols*8-1:0]    b_col_i,

  // C, one column per accepted B column, Lat cycles later
  output logic                 c_valid_o,
  output logic [Rows*AccW-1:0] c_col_o
);

  localparam int unsigned Lat   = Rows + Cols + 1;
  localparam int unsigned CntW  = $clog2(Rows + 1);
  localparam int unsigned HoldW = (Rows + Cols > 2) ? $clog2(Rows + Cols - 1) : 1;

`ifndef SYNTHESIS
  initial begin
    if (Rows < 1 || Cols < 1)
      $fatal(1, "bpu_sa_array: Rows and Cols must be at least 1");
    if (AccW < 16 + $clog2(Cols))
      $fatal(1, "bpu_sa_array: AccW = %0d is below 16 + clog2(Cols) = %0d",
             AccW, 16 + $clog2(Cols));
  end
`endif

  assign err_mode_o = mode_i;
  assign b_ready_o  = 1'b1;

  // ---------------------------------------------------------------------------
  // Control: shadow fill count, tile switch, reload hold-off
  // ---------------------------------------------------------------------------
  logic [CntW-1:0]  a_cnt_q;   // rows of the next tile in the shadow registers
  logic [HoldW-1:0] hold_q;    // cycles until the last tile switch has reached every PE
  logic             a_fire, swap;

  assign a_full_o  = (a_cnt_q == CntW'(Rows));
  assign a_ready_o = !a_full_o && (hold_q == '0);
  assign a_fire    = a_valid_i && a_ready_o;
  assign swap      = b_valid_i && a_full_o;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      a_cnt_q <= '0;
      hold_q  <= '0;
    end else if (swap) begin
      a_cnt_q <= '0;
      hold_q  <= HoldW'(Rows + Cols - 2);
    end else begin
      if (a_fire)        a_cnt_q <= a_cnt_q + 1'b1;
      if (hold_q != '0)  hold_q  <= hold_q - 1'b1;
    end
  end

  // ---------------------------------------------------------------------------
  // B input register and skew: column k is delayed k cycles
  // ---------------------------------------------------------------------------
  logic [Cols-1:0][7:0] b_q, top_b;
  logic                 swap_q;
  logic [Cols-1:0]      top_swap;

  always_ff @(posedge clk_i) begin
    b_q <= b_valid_i ? b_col_i : '0;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) swap_q <= 1'b0;
    else         swap_q <= swap;
  end

  for (genvar k = 0; k < Cols; k++) begin : g_skew
    bpu_sa_delay #(.Width(8), .Depth(k)) u_b (
      .clk_i, .rst_ni, .d_i(b_q[k]), .q_o(top_b[k])
    );
    bpu_sa_delay #(.Width(1), .Depth(k), .Reset(1'b1)) u_swap (
      .clk_i, .rst_ni, .d_i(swap_q), .q_o(top_swap[k])
    );
  end

  // ---------------------------------------------------------------------------
  // PE grid. Arrays hold each PE's outputs; inputs come from the neighbours.
  // ---------------------------------------------------------------------------
  logic [Cols-1:0][7:0]                a_top;
  logic [Rows-1:0][Cols-1:0][7:0]      a_dn, b_dn;
  logic [Rows-1:0][Cols-1:0]           swap_dn;
  logic [Rows-1:0][Cols-1:0][AccW-1:0] sum_rt, carry_rt;

  assign a_top = a_row_i;

  for (genvar i = 0; i < Rows; i++) begin : g_row
    for (genvar k = 0; k < Cols; k++) begin : g_col
      logic [7:0]      a_in, b_in;
      logic            swap_in;
      logic [AccW-1:0] sum_in, carry_in;

      if (i == 0) begin : g_top
        assign a_in    = a_top[k];
        assign b_in    = top_b[k];
        assign swap_in = top_swap[k];
      end else begin : g_below
        assign a_in    = a_dn[i-1][k];
        assign b_in    = b_dn[i-1][k];
        assign swap_in = swap_dn[i-1][k];
      end

      if (k == 0) begin : g_left
        assign sum_in   = '0;
        assign carry_in = '0;
      end else begin : g_right
        assign sum_in   = sum_rt[i][k-1];
        assign carry_in = carry_rt[i][k-1];
      end

      bpu_sa_pe #(.AccW(AccW)) u_pe (
        .clk_i,
        .rst_ni,
        .mode_i,
        .load_i   (a_fire),
        .load_a_i (a_in),
        .load_a_o (a_dn[i][k]),
        .b_i      (b_in),
        .swap_i   (swap_in),
        .b_o      (b_dn[i][k]),
        .swap_o   (swap_dn[i][k]),
        .sum_i    (sum_in),
        .carry_i  (carry_in),
        .sum_o    (sum_rt[i][k]),
        .carry_o  (carry_rt[i][k])
      );
    end
  end

  // ---------------------------------------------------------------------------
  // Row edge: resolve sum + carry, deskew (row i is i cycles later than row 0),
  // register the whole column
  // ---------------------------------------------------------------------------
  logic [Rows-1:0][AccW-1:0] row_c, row_al, c_q;

  for (genvar i = 0; i < Rows; i++) begin : g_edge
    assign row_c[i] = sum_rt[i][Cols-1] + carry_rt[i][Cols-1];
    bpu_sa_delay #(.Width(AccW), .Depth(Rows - 1 - i)) u_deskew (
      .clk_i, .rst_ni, .d_i(row_c[i]), .q_o(row_al[i])
    );
  end

  always_ff @(posedge clk_i) begin
    c_q <= row_al;
  end

  assign c_col_o = c_q;

  bpu_sa_delay #(.Width(1), .Depth(Lat), .Reset(1'b1)) u_valid (
    .clk_i, .rst_ni, .d_i(b_valid_i), .q_o(c_valid_o)
  );

endmodule
