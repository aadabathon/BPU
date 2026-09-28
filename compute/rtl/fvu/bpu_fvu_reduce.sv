// FVU row reductions (RSUM, RDOT, RMAX, RAMAX), bit-exact with bpuref.fvu.
//
// Input: one item per word of a row, including the +0 padding words that extend
// the row to P' = max(64, next_pow2(cols)) elements; masked lanes count as +0
// (sums) or are ignored (max).
//   sum: a lane adder tree forms the word's node of the canonical tree; a merge
//        stack then combines word nodes pairwise (a binary counter over levels),
//        one fp32 add at a time, exactly as the canonical tree does.
//   max: running maximum under the total-order key; all-NaN gives canonical NaN.
// Each finished row produces one element write (wr_*). pop_o returns a credit to
// the issuing sequencer for every item taken out of the input FIFO.
module bpu_fvu_reduce #(
  parameter int unsigned Lanes     = 4,
  parameter int unsigned AW        = 16,       // element address width
  parameter logic [2:0]  AddPipe   = 3'b111,
  parameter int unsigned FifoDepth = 8
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
  localparam int unsigned LMax = 18;                       // stack levels (rows up to 2^17 words)
  localparam int unsigned LvW  = $clog2(LMax + 1);

  // ---------------------------------------------------------------------------
  // Word stage: lane tree (sum) and lane max
  // ---------------------------------------------------------------------------
  logic [Lanes*32-1:0] sum_in;
  logic [31:0]         node_sum;
  logic [31:0]         wkey, wval;

  for (genvar l = 0; l < Lanes; l++) begin : g_mask
    assign sum_in[l*32 +: 32] = in_mask_i[l] ? in_vals_i[l*32 +: 32] : 32'd0;
  end

  bpu_fvu_sumtree #(.N(Lanes), .AddPipe(AddPipe)) u_tree (
    .clk_i, .rst_ni, .in_i(sum_in), .sum_o(node_sum));

  always_comb begin
    wkey = '0;
    wval = Fp32QNaN;
    for (int l = 0; l < Lanes; l++) begin
      if (in_mask_i[l] && f32_order_key(in_vals_i[l*32 +: 32]) > wkey) begin
        wkey = f32_order_key(in_vals_i[l*32 +: 32]);
        wval = in_vals_i[l*32 +: 32];
      end
    end
  end

  localparam int unsigned MetaW = 3 + AW + 64;
  logic             t_valid;
  logic             t_max, t_first, t_last;
  logic [AW-1:0]    t_dest;
  logic [31:0]      t_key, t_val;

  bpu_delay #(.Width(1), .Depth(Lt), .Reset(1'b1)) u_dv (
    .clk_i, .rst_ni, .d_i(in_valid_i), .q_o(t_valid));
  bpu_delay #(.Width(MetaW), .Depth(Lt)) u_dm (
    .clk_i, .rst_ni, .d_i({in_max_i, in_first_i, in_last_i, in_dest_i, wkey, wval}),
    .q_o({t_max, t_first, t_last, t_dest, t_key, t_val}));

  // Items inside the tree pipeline (for busy_o).
  logic [$clog2(Lt+2)-1:0] tree_cnt_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) tree_cnt_q <= '0;
    else         tree_cnt_q <= tree_cnt_q + ($bits(tree_cnt_q))'(in_valid_i) - ($bits(tree_cnt_q))'(t_valid);
  end

  // ---------------------------------------------------------------------------
  // Input FIFO
  // ---------------------------------------------------------------------------
  localparam int unsigned FW = 3 + AW + 96;
  logic          f_valid, f_pop, fifo_ready;
  logic          f_max, f_first, f_last;
  logic [AW-1:0] f_dest;
  logic [31:0]   f_sum, f_key, f_val;

  bpu_fifo #(.Width(FW), .Depth(FifoDepth)) u_fifo (
    .clk_i, .rst_ni,
    .in_valid_i(t_valid), .in_ready_o(fifo_ready),
    .in_data_i({t_max, t_first, t_last, t_dest, node_sum, t_key, t_val}),
    .out_valid_o(f_valid), .out_ready_i(f_pop),
    .out_data_o({f_max, f_first, f_last, f_dest, f_sum, f_key, f_val}),
    .count_o()
  );
  assign pop_o = f_pop;

  // ---------------------------------------------------------------------------
  // Merge FSM
  // ---------------------------------------------------------------------------
  typedef enum logic [1:0] {MIdle, MCarry, MWait} mstate_e;
  mstate_e          st_q;
  logic [31:0]      cur_q, mkey_q, mval_q;
  logic [LvW-1:0]   lvl_q;
  logic [LMax-1:0]  occ_q;
  logic [LMax-1:0][31:0] stack_q;         // packed: registers, never an SRAM
  logic             last_q;
  logic [AW-1:0]    dest_q;
  localparam int unsigned CntW = (La > 0) ? $clog2(La + 1) : 1;
  logic [CntW-1:0] cnt_q;
  logic [31:0]      add_y;
  logic             add_go;

  bpu_fp32_add #(.PipeMask(AddPipe)) u_merge_add (
    .clk_i, .rst_ni, .valid_i(add_go), .a_i(stack_q[lvl_q]), .b_i(cur_q), .valid_o(), .y_o(add_y));

  // Max path (single cycle).
  logic        mx_better;
  logic [31:0] nkey, nval;
  assign mx_better = f_first || (f_key > mkey_q);
  assign nkey      = mx_better ? f_key : mkey_q;
  assign nval      = mx_better ? f_val : mval_q;

  assign f_pop  = (st_q == MIdle) && f_valid;
  assign add_go = (st_q == MCarry) && occ_q[lvl_q];

  always_comb begin
    wr_valid_o = 1'b0;
    wr_addr_o  = dest_q;
    wr_data_o  = cur_q;
    if (st_q == MIdle && f_valid && f_max && f_last) begin
      wr_valid_o = 1'b1;
      wr_addr_o  = f_dest;
      wr_data_o  = (nkey == '0) ? Fp32QNaN : nval;
    end else if (st_q == MCarry && !occ_q[lvl_q] && last_q) begin
      wr_valid_o = 1'b1;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      st_q  <= MIdle;
      occ_q <= '0;
      lvl_q <= '0;
      cnt_q <= '0;
    end else begin
      unique case (st_q)
        MIdle: begin
          if (f_valid && !f_max) begin
            st_q  <= MCarry;
            lvl_q <= '0;
          end
        end
        MCarry: begin
          if (occ_q[lvl_q]) begin
            occ_q[lvl_q] <= 1'b0;
            lvl_q        <= lvl_q + 1'b1;
            if (La != 0) begin
              st_q  <= MWait;
              cnt_q <= ($bits(cnt_q))'(La - 1);
            end
          end else begin
            occ_q[lvl_q] <= 1'b1;
            if (last_q) occ_q <= '0;          // the row's root is out; stack is empty
            st_q <= MIdle;
          end
        end
        MWait: begin
          if (cnt_q == '0) st_q <= MCarry;
          else             cnt_q <= cnt_q - 1'b1;
        end
        default: st_q <= MIdle;
      endcase
    end
  end

  always_ff @(posedge clk_i) begin
    if (f_pop) begin
      if (f_max) begin
        mkey_q <= nkey;
        mval_q <= nval;
      end else begin
        cur_q  <= f_sum;
        last_q <= f_last;
        dest_q <= f_dest;
      end
    end
    if (st_q == MCarry && !occ_q[lvl_q]) stack_q[lvl_q] <= cur_q;
    // The merge result lands in cur_q: same cycle (combinational adder) or after the wait.
    if ((st_q == MCarry && occ_q[lvl_q] && La == 0) || (st_q == MWait && cnt_q == '0))
      cur_q <= add_y;
  end

  assign busy_o = (tree_cnt_q != '0) || f_valid || (st_q != MIdle);

`ifdef FORMAL
  always_comb begin
    if (rst_ni) begin
      assert (!(t_valid && !fifo_ready));           // credits keep the FIFO from overflowing
      assert (lvl_q <= LvW'(LMax));
    end
  end
`endif

`ifdef SYNTHESIS
  logic unused_fifo_ready;
  assign unused_fifo_ready = fifo_ready;
`else
  always @(posedge clk_i) begin
    if (rst_ni && t_valid && !fifo_ready)
      $error("bpu_fvu_reduce: input FIFO overflow (credit accounting broken)");
  end
`endif

endmodule
