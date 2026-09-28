// FVU row reductions (RSUM, RDOT, RMAX, RAMAX), bit-exact with bpuref.fvu.
//
// Input: one item per word of a row, including the +0 padding words that extend
// the row to P' = max(64, next_pow2(cols)) elements; masked lanes count as +0
// (sums) or are ignored (max).
//   sum: a lane adder tree forms the word's node of the canonical tree; a
//        pipelined merge then combines nodes pairwise level by level, exactly as
//        the canonical tree does, taking up to one word node per cycle.
//   max: running maximum of the total-order key; all-NaN gives canonical NaN.
// Each finished row produces one element write (wr_*). pop_o returns a credit to
// the issuing sequencer for every item taken out of the input FIFO.
module bpu_fvu_reduce #(
  parameter int unsigned Lanes     = 4,
  parameter int unsigned AW        = 16,       // element address width
  parameter logic [2:0]  AddPipe   = 3'b111,
  parameter int unsigned FifoDepth = 8,
  parameter int unsigned MaxColsLog2 = 16      // rows up to 2^16 elements (the cols field)
) (
  input  logic               clk_i,
  input  logic               rst_ni,

  input  logic               in_valid_i,
  input  logic               in_max_i,     // max reduction (else sum)
  input  logic               in_first_i,   // first word of a row
  input  logic               in_last_i,    // last (padded) word of a row
  input  logic [AW-1:0]      in_dest_i,    // element address of the row result
  input  logic [Lanes-1:0]   in_mask_i,
  input  logic [Lanes*32-1:0] in_vals_i,

  output logic               pop_o,

  output logic               wr_valid_o,
  output logic [AW-1:0]      wr_addr_o,
  output logic [31:0]        wr_data_o,

  output logic               busy_o
);

  import bpu_compute_pkg::*;

  localparam int unsigned La   = pipe3_latency(AddPipe);
  localparam int unsigned Lt   = $clog2(Lanes) * La;

  // ---------------------------------------------------------------------------
  // Word stage: lane tree (sum) and lane max
  // ---------------------------------------------------------------------------
  logic [Lanes*32-1:0] sum_in;
  logic [31:0]         node_sum;
  logic [31:0]         wkey;

  for (genvar l = 0; l < Lanes; l++) begin : g_mask
    assign sum_in[l*32 +: 32] = in_mask_i[l] ? in_vals_i[l*32 +: 32] : 32'd0;
  end

  bpu_fvu_sumtree #(.N(Lanes), .AddPipe(AddPipe)) u_tree (
    .clk_i, .rst_ni, .in_i(sum_in), .sum_o(node_sum));

  // Word max as a balanced tree over order keys. Equal keys mean equal bits and
  // the key is invertible, so only the key travels on (0: masked or NaN).
  logic [Lanes-1:0][31:0] mk;
  always_comb begin
    for (int l = 0; l < Lanes; l++)
      mk[l] = in_mask_i[l] ? f32_order_key(in_vals_i[l*32 +: 32]) : 32'd0;
    for (int s = 1; s < Lanes; s = s * 2)
      for (int l = 0; l + s < Lanes; l = l + 2 * s)
        if (mk[l+s] > mk[l]) mk[l] = mk[l+s];
    wkey = mk[0];
  end

  logic             t_valid;
  logic             t_max, t_first, t_last;
  logic [AW-1:0]    t_dest;
  logic [31:0]      t_key;

  bpu_delay #(.Width(1), .Depth(Lt), .Reset(1'b1)) u_dv (
    .clk_i, .rst_ni, .d_i(in_valid_i), .q_o(t_valid));
  bpu_delay #(.Width(3 + AW + 32), .Depth(Lt)) u_dm (
    .clk_i, .rst_ni, .d_i({in_max_i, in_first_i, in_last_i, in_dest_i, wkey}),
    .q_o({t_max, t_first, t_last, t_dest, t_key}));

  // Items inside the tree pipeline (for busy_o).
  logic [$clog2(Lt+2)-1:0] tree_cnt_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) tree_cnt_q <= '0;
    else         tree_cnt_q <= tree_cnt_q + ($bits(tree_cnt_q))'(in_valid_i) - ($bits(tree_cnt_q))'(t_valid);
  end

  // ---------------------------------------------------------------------------
  // Input FIFO: one 32-bit value per word (sum node, or max key)
  // ---------------------------------------------------------------------------
  logic          f_valid, f_pop, fifo_ready;
  logic          f_max, f_first, f_last;
  logic [AW-1:0] f_dest;
  logic [31:0]   f_node;

  bpu_fifo #(.Width(3 + AW + 32), .Depth(FifoDepth)) u_fifo (
    .clk_i, .rst_ni,
    .in_valid_i(t_valid), .in_ready_o(fifo_ready),
    .in_data_i({t_max, t_first, t_last, t_dest, t_max ? t_key : node_sum}),
    .out_valid_o(f_valid), .out_ready_i(f_pop),
    .out_data_o({f_max, f_first, f_last, f_dest, f_node}),
    .count_o()
  );
  assign pop_o = f_pop;

  // ---------------------------------------------------------------------------
  // Pipelined merge
  // ---------------------------------------------------------------------------
  // Level-L nodes of the canonical tree are paired left/right in arrival order:
  // pend_q[L] holds a left node until its right sibling arrives, then one add
  // forms the level-(L+1) node. Word nodes (level 0) come from the FIFO; higher
  // nodes come back from the pipelined adder, tagged with level, last flag and
  // destination. The adder has a fixed latency, so every level's nodes arrive
  // in order and the pairing is exactly the canonical adjacent-pair tree, while
  // adds from different levels and rows overlap in the adder pipeline.
  // A row's rightmost node at a level with no pending left sibling is the root.
  // A completed word pair waits in a one-entry pair buffer, so FIFO pops keep
  // flowing while returning nodes (which have priority) hold the adder.
  localparam int unsigned NLvl = (MaxColsLog2 > $clog2(Lanes)) ? MaxColsLog2 - $clog2(Lanes) : 1;
  localparam int unsigned LvW  = $clog2(NLvl + 1);
  localparam int unsigned LiW  = (NLvl > 1) ? $clog2(NLvl) : 1;
  localparam int unsigned Le   = (La > 0) ? La : 1;     // result latency (registered if La == 0)

  logic [NLvl-1:0]        pend_v_q;
  logic [NLvl-1:0][31:0]  pend_q;          // packed: registers, never an SRAM

  // Node returning from the adder
  logic           r_v, r_last;
  logic [LvW-1:0] r_lvl;
  logic [LiW-1:0] r_idx;
  logic [AW-1:0]  r_dest;
  logic [31:0]    r_val, add_y;

  // Adder issue
  logic           add_go, g_last;
  logic [LvW-1:0] g_lvl;
  logic [AW-1:0]  g_dest;
  logic [31:0]    g_a, g_b;

  // Pair buffer: a word pair waiting for the adder
  logic           pair_v_q, pair_last_q, pair_go;
  logic [AW-1:0]  pair_dest_q;
  logic [31:0]    pair_a_q, pair_b_q;

  logic r_in, r_pair, r_root, r_keep;
  assign r_idx  = r_lvl[LiW-1:0];
  assign r_in   = (32'(r_lvl) < NLvl) && pend_v_q[r_idx];     // left sibling waiting
  assign r_pair = r_v &&  r_in;                               // right sibling: merge
  assign r_root = r_v && !r_in &&  r_last;                    // row result
  assign r_keep = r_v && !r_in && !r_last;                    // left node: park

  // FIFO head
  logic f_is_sum, f_pair, f_root, f_mroot, f_keep;
  assign f_is_sum = f_valid && !f_max;
  assign f_pair   = f_is_sum && pend_v_q[0];
  assign f_root   = f_is_sum && !pend_v_q[0] && f_last;       // one-word row
  assign f_keep   = f_is_sum && !pend_v_q[0] && !f_last;
  assign f_mroot  = f_valid && f_max && f_last;

  assign pair_go = pair_v_q && !r_pair;
  assign f_pop   = f_valid && !(f_pair && pair_v_q && !pair_go)
                           && !((f_root || f_mroot) && r_root);

  assign add_go = r_pair || pair_go;
  assign g_a    = r_pair ? pend_q[r_idx] : pair_a_q;
  assign g_b    = r_pair ? r_val : pair_b_q;
  assign g_lvl  = r_pair ? r_lvl + 1'b1 : LvW'(1);
  assign g_last = r_pair ? r_last : pair_last_q;
  assign g_dest = r_pair ? r_dest : pair_dest_q;

  bpu_fp32_add #(.PipeMask(AddPipe)) u_merge_add (
    .clk_i, .rst_ni, .valid_i(add_go), .a_i(g_a), .b_i(g_b), .valid_o(), .y_o(add_y));
  bpu_delay #(.Width(32), .Depth(Le - La)) u_rv (
    .clk_i, .rst_ni, .d_i(add_y), .q_o(r_val));
  bpu_delay #(.Width(1), .Depth(Le), .Reset(1'b1)) u_tv (
    .clk_i, .rst_ni, .d_i(add_go), .q_o(r_v));
  bpu_delay #(.Width(LvW + 1 + AW), .Depth(Le)) u_tm (
    .clk_i, .rst_ni, .d_i({g_lvl, g_last, g_dest}), .q_o({r_lvl, r_last, r_dest}));

  // Adds in flight (for busy_o).
  logic [$clog2(Le+2)-1:0] add_cnt_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) add_cnt_q <= '0;
    else         add_cnt_q <= add_cnt_q + ($bits(add_cnt_q))'(add_go) - ($bits(add_cnt_q))'(r_v);
  end

  // Max path (single cycle): running maximum key.
  logic [31:0] mkey_q, nkey;
  assign nkey = (f_first || f_node > mkey_q) ? f_node : mkey_q;

  always_comb begin
    wr_valid_o = 1'b0;
    wr_addr_o  = r_dest;
    wr_data_o  = r_val;
    if (r_root) begin
      wr_valid_o = 1'b1;
    end else if (f_pop && f_root) begin
      wr_valid_o = 1'b1;
      wr_addr_o  = f_dest;
      wr_data_o  = f_node;
    end else if (f_pop && f_mroot) begin
      wr_valid_o = 1'b1;
      wr_addr_o  = f_dest;
      wr_data_o  = f32_from_order_key(nkey);
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      pend_v_q <= '0;
      pair_v_q <= 1'b0;
    end else begin
      if (r_pair) pend_v_q[r_idx] <= 1'b0;
      if (r_keep) pend_v_q[r_idx] <= 1'b1;
      if (f_pop && f_pair) pend_v_q[0] <= 1'b0;
      if (f_pop && f_keep) pend_v_q[0] <= 1'b1;
      if (f_pop && f_pair) pair_v_q <= 1'b1;
      else if (pair_go)    pair_v_q <= 1'b0;
    end
  end

  always_ff @(posedge clk_i) begin
    if (r_keep)          pend_q[r_idx] <= r_val;
    if (f_pop && f_keep) pend_q[0]     <= f_node;
    if (f_pop && f_pair) begin
      pair_a_q    <= pend_q[0];
      pair_b_q    <= f_node;
      pair_last_q <= f_last;
      pair_dest_q <= f_dest;
    end
    if (f_pop && f_max)  mkey_q        <= nkey;
  end

  assign busy_o = (tree_cnt_q != '0) || f_valid || pair_v_q || (add_cnt_q != '0) || (pend_v_q != '0);

`ifdef FORMAL
  always_comb begin
    if (rst_ni) begin
      assert (!(t_valid && !fifo_ready));           // credits keep the FIFO from overflowing
      if (r_v) assert (r_lvl != '0);                // returning nodes never touch level 0
      if (r_keep) assert (32'(r_lvl) < NLvl);       // a parked node always has a register
      assert (add_cnt_q <= ($bits(add_cnt_q))'(Le));
    end
  end
`endif

`ifdef SYNTHESIS
  logic unused_fifo_ready;
  assign unused_fifo_ready = fifo_ready;
`else
  /* verilator lint_off SYNCASYNCNET */
  always @(posedge clk_i) begin
    if (rst_ni && t_valid && !fifo_ready)
      $error("bpu_fvu_reduce: input FIFO overflow (credit accounting broken)");
  end
  /* verilator lint_on SYNCASYNCNET */
`endif

endmodule
