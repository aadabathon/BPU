// Command sequencer: accepts descriptors in program order, routes them to one queue
// per unit, and issues each unit's oldest command once the commands it waits for
// have completed. Units run concurrently, one command in flight per unit.
//
// Descriptor: {unit, tag, wait mask, body} (layout: bpu_isa_pkg).
//   tag   names the command in the scoreboard. A descriptor is not accepted while
//         an earlier command with the same tag is still pending, so software may
//         reuse tags freely.
//   wait  tags of earlier commands this one depends on. The mask is resolved at
//         acceptance against the commands pending at that moment and then only
//         shrinks as they complete, so a dependency always means a specific earlier
//         command and never a later reuse of its tag. Bits of tags that are not
//         pending (already complete) and the command's own tag are ignored.
//   unit  0 vector, 1 matrix, 2 memory manager; 3 is reserved and completes at
//         once with an error.
// Commands to the same unit execute in order, so software only needs wait bits for
// dependencies on other units. Completion (cpl_o) of a command means its results
// are visible to every later command.
module bpu_cmd_seq #(
  parameter int unsigned NTags  = 16,
  parameter int unsigned QDepth = 2,
  parameter int unsigned BodyW  = 432,
  // Body bits each unit's queue keeps (the rest of the body is dropped)
  parameter int unsigned VecBodyW = BodyW,
  parameter int unsigned MatBodyW = BodyW,
  parameter int unsigned MemBodyW = BodyW
) (
  input  logic                        clk_i,
  input  logic                        rst_ni,

  input  logic                        desc_valid_i,
  output logic                        desc_ready_o,
  input  logic [1:0]                  desc_unit_i,
  input  logic [$clog2(NTags)-1:0]    desc_tag_i,
  input  logic [NTags-1:0]            desc_wait_i,
  input  logic [BodyW-1:0]            desc_body_i,

  // Units: 0 vector, 1 matrix, 2 memory manager
  output logic [2:0]                  iss_valid_o,
  input  logic [2:0]                  iss_ready_i,
  output logic [3*BodyW-1:0]          iss_body_o,
  input  logic [2:0]                  done_i,
  input  logic [2:0]                  done_err_i,

  // Scoreboard
  output logic [NTags-1:0]            pending_o,
  output logic [NTags-1:0]            err_o,
  output logic [NTags-1:0]            cpl_o,        // completions this cycle
  output logic                        busy_o,
  output logic [31:0]                 perf_accept_wait_o    // cycles a descriptor waited
);

  localparam int unsigned TW = $clog2(NTags);
  localparam int unsigned QW = (QDepth > 1) ? $clog2(QDepth) : 1;

  logic [NTags-1:0] pending, cpl, cpl_err;
  assign pending_o = pending;
  assign cpl_o     = cpl;

  // ---------------------------------------------------------------------------
  // Unit queues
  // ---------------------------------------------------------------------------
  logic [2:0][QDepth-1:0]             qv_q;
  logic [2:0][QDepth-1:0][TW-1:0]     qtag_q;
  logic [2:0][QDepth-1:0][NTags-1:0]  qrem_q;
  logic [2:0][QW-1:0]                 qhead_q, qtail_q;
  logic [2:0]                         qfull;

  logic [2:0]          run_q;
  logic [2:0][TW-1:0]  run_tag_q;

  for (genvar u = 0; u < 3; u++) begin : g_qfull
    assign qfull[u] = qv_q[u][qtail_q[u]];
  end

  // Acceptance
  logic accept, bad;
  logic [NTags-1:0] own, rem_new;
  assign own     = NTags'(1) << desc_tag_i;
  assign rem_new = desc_wait_i & pending & ~cpl & ~own;
  logic [3:0] qfull4;
  assign qfull4       = {1'b0, qfull};                // unit 3 never waits for a queue
  assign desc_ready_o = !pending[desc_tag_i] && !qfull4[desc_unit_i];
  assign accept  = desc_valid_i && desc_ready_o;
  assign bad     = accept && desc_unit_i == 2'd3;

  // Issue
  logic [2:0] issue;
  for (genvar u = 0; u < 3; u++) begin : g_iss
    assign iss_valid_o[u] = qv_q[u][qhead_q[u]] && !run_q[u] && qrem_q[u][qhead_q[u]] == '0;
    assign issue[u] = iss_valid_o[u] && iss_ready_i[u];
  end

  // Bodies: each unit's queue keeps only the bits that unit uses.
  for (genvar u = 0; u < 3; u++) begin : g_body
    localparam int unsigned UW = (u == 0) ? VecBodyW : (u == 1) ? MatBodyW : MemBodyW;
    logic [QDepth-1:0][UW-1:0] body_q;
    always_ff @(posedge clk_i) begin
      if (accept && desc_unit_i == 2'(u)) body_q[qtail_q[u]] <= desc_body_i[UW-1:0];
    end
    assign iss_body_o[u*BodyW +: BodyW] = BodyW'(body_q[qhead_q[u]]);
    if (UW < BodyW) begin : g_unused
      logic unused_hi;
      assign unused_hi = ^desc_body_i[BodyW-1:UW];
    end
  end

  // Completions: the running command of each unit, plus a reserved-unit descriptor
  logic          bad_q;
  logic [TW-1:0] bad_tag_q;
  always_comb begin
    cpl     = '0;
    cpl_err = '0;
    for (int u = 0; u < 3; u++) begin
      if (done_i[u] && run_q[u]) begin
        cpl[run_tag_q[u]]     = 1'b1;
        cpl_err[run_tag_q[u]] = done_err_i[u];
      end
    end
    if (bad_q) begin
      cpl[bad_tag_q]     = 1'b1;
      cpl_err[bad_tag_q] = 1'b1;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      qv_q    <= '0;
      qhead_q <= '0;
      qtail_q <= '0;
      run_q   <= '0;
      bad_q   <= 1'b0;
    end else begin
      bad_q <= bad;
      for (int u = 0; u < 3; u++) begin
        if (issue[u]) begin
          qv_q[u][qhead_q[u]] <= 1'b0;
          qhead_q[u]          <= (int'(qhead_q[u]) == QDepth - 1) ? '0 : qhead_q[u] + 1'b1;
        end
        if (accept && !bad && desc_unit_i == 2'(u)) begin
          qv_q[u][qtail_q[u]] <= 1'b1;
          qtail_q[u]          <= (int'(qtail_q[u]) == QDepth - 1) ? '0 : qtail_q[u] + 1'b1;
        end
        if (done_i[u]) run_q[u] <= 1'b0;
        if (issue[u])  run_q[u] <= 1'b1;
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (bad) bad_tag_q <= desc_tag_i;
    for (int u = 0; u < 3; u++) begin
      if (issue[u]) run_tag_q[u] <= qtag_q[u][qhead_q[u]];
      for (int e = 0; e < QDepth; e++) qrem_q[u][e] <= qrem_q[u][e] & ~cpl;
      if (accept && !bad && desc_unit_i == 2'(u)) begin
        qtag_q[u][qtail_q[u]]  <= desc_tag_i;
        qrem_q[u][qtail_q[u]]  <= rem_new;
      end
    end
  end

  bpu_scoreboard #(.NTags(NTags)) u_sb (
    .clk_i, .rst_ni, .set_i(accept), .set_tag_i(desc_tag_i), .cpl_i(cpl), .cpl_err_i(cpl_err),
    .pending_o(pending), .err_o);

  assign busy_o = (pending != '0);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) perf_accept_wait_o <= '0;
    else         perf_accept_wait_o <= perf_accept_wait_o + 32'(desc_valid_i && !desc_ready_o);
  end

`ifdef FORMAL
  always_comb begin
    if (rst_ni) begin
      for (int u = 0; u < 3; u++) begin
        // Every queued or running command is pending in the scoreboard.
        for (int e = 0; e < QDepth; e++) begin
          if (qv_q[u][e]) assert (pending[qtag_q[u][e]]);
          // Remaining dependencies are always pending, never the command itself.
          if (qv_q[u][e]) assert ((qrem_q[u][e] & ~pending) == '0 && !qrem_q[u][e][qtag_q[u][e]]);
        end
        if (run_q[u]) assert (pending[run_tag_q[u]]);
        // Issue only with no outstanding dependency and the unit idle.
        if (iss_valid_o[u]) assert (!run_q[u] && qrem_q[u][qhead_q[u]] == '0);
      end
    end
  end
`endif

endmodule
