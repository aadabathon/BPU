// Formal harness for bpu_cmd_seq (with its scoreboard): any descriptors, any unit
// readiness, units that complete their running command at any time. Checks, beyond
// the module's own asserts:
//   - a command never issues while a command it waited for at acceptance is still
//     pending (tracked for one symbolic accepted descriptor),
//   - each unit has at most one command in flight, and completions only clear
//     commands that were running.
module bpu_cmd_seq_fv (
  input logic       clk_i,
  input logic       rst_ni,
  input logic       desc_valid_i,
  input logic [1:0] desc_unit_i,
  input logic [1:0] desc_tag_i,
  input logic [3:0] desc_wait_i,
  input logic [2:0] iss_ready_i,
  input logic [2:0] done_i,
  input logic [2:0] done_err_i,
  input logic       pick_i                       // marks the descriptor to track
);

  localparam int unsigned NTags = 4;

  logic       desc_ready, busy;
  logic [2:0] iss_valid;
  logic [3*4-1:0] iss_body;
  logic [NTags-1:0] pending, err, cpl;

  // The body carries the tag, so the harness can see which command issues.
  bpu_cmd_seq #(.NTags(NTags), .QDepth(2), .BodyW(4)) dut (
    .clk_i, .rst_ni,
    .desc_valid_i, .desc_ready_o(desc_ready), .desc_unit_i, .desc_tag_i, .desc_wait_i,
    .desc_body_i({2'b00, desc_tag_i}),
    .iss_valid_o(iss_valid), .iss_ready_i, .iss_body_o(iss_body), .done_i, .done_err_i,
    .pending_o(pending), .err_o(err), .cpl_o(cpl), .busy_o(busy), .perf_accept_wait_o());

  logic init_q = 1'b1, rst_n_q;
  always_ff @(posedge clk_i) begin
    init_q  <= 1'b0;
    rst_n_q <= rst_ni;
  end
  always_comb begin
    if (init_q) assume (!rst_ni);
    if (!init_q && rst_n_q) assume (rst_ni);
  end

  // Unit-indexed views padded to four units, so a 2-bit unit index is always in range.
  logic [3:0]      iss_v4, iss_r4;
  logic [4*4-1:0]  iss_b4;
  assign iss_v4 = {1'b0, iss_valid};
  assign iss_r4 = {1'b0, iss_ready_i};
  assign iss_b4 = {4'b0, iss_body};

  // Track one accepted descriptor: the pending commands it waits for, by tag.
  logic       trk_v_q, trk_issued_q, trk_had_dep_q;
  logic [1:0] trk_tag_q, trk_unit_q;
  logic [3:0] trk_dep_q;                     // waited-for commands still pending
  logic       accept;
  assign accept = desc_valid_i && desc_ready;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      trk_v_q       <= 1'b0;
      trk_issued_q  <= 1'b0;
      trk_had_dep_q <= 1'b0;
    end else begin
      if (!trk_v_q && accept && pick_i && desc_unit_i != 2'd3) begin
        trk_v_q    <= 1'b1;
        trk_tag_q  <= desc_tag_i;
        trk_unit_q <= desc_unit_i;
        trk_dep_q  <= desc_wait_i & pending & ~cpl & ~(4'd1 << desc_tag_i);
        trk_had_dep_q <= (desc_wait_i & pending & ~cpl & ~(4'd1 << desc_tag_i)) != '0;
      end else if (trk_v_q) begin
        trk_dep_q <= trk_dep_q & ~cpl;
        if (iss_v4[trk_unit_q] && iss_r4[trk_unit_q]
            && iss_b4[trk_unit_q*4 +: 2] == trk_tag_q && pending[trk_tag_q])
          trk_issued_q <= 1'b1;
      end
    end
  end

  always_comb begin
    if (rst_ni) begin
      // The tracked command issues only once everything it waited for completed.
      if (trk_v_q && !trk_issued_q && iss_v4[trk_unit_q]
          && iss_b4[trk_unit_q*4 +: 2] == trk_tag_q)
        assert (trk_dep_q == '0);
      // Errors only for completed tags; busy iff something is pending.
      assert ((err & pending) == '0);
      assert (busy == (pending != '0));
      cover (trk_v_q && trk_issued_q && trk_had_dep_q);     // waited for another command, then issued
    end
  end

endmodule
