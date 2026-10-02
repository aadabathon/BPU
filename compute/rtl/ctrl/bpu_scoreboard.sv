// Scoreboard: one pending bit and one error bit per command tag.
// A tag is set pending when its command is accepted and cleared when the command
// completes; the error bit holds the completion status until the tag is reused.
module bpu_scoreboard #(
  parameter int unsigned NTags = 16
) (
  input  logic              clk_i,
  input  logic              rst_ni,
  input  logic              set_i,                 // command accepted
  input  logic [$clog2(NTags)-1:0] set_tag_i,
  input  logic [NTags-1:0]  cpl_i,                 // commands completing this cycle
  input  logic [NTags-1:0]  cpl_err_i,
  output logic [NTags-1:0]  pending_o,
  output logic [NTags-1:0]  err_o
);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      pending_o <= '0;
      err_o     <= '0;
    end else begin
      for (int t = 0; t < NTags; t++) begin
        if (set_i && set_tag_i == ($clog2(NTags))'(t)) begin
          pending_o[t] <= 1'b1;
          err_o[t]     <= 1'b0;
        end else if (cpl_i[t]) begin
          pending_o[t] <= 1'b0;
          err_o[t]     <= cpl_err_i[t];
        end
      end
    end
  end

`ifdef FORMAL
  always_comb begin
    if (rst_ni) begin
      // A tag is never set while pending, and only pending tags complete.
      if (set_i) assert (!pending_o[set_tag_i]);
      assert ((cpl_i & ~pending_o) == '0);
    end
  end
`endif

endmodule
