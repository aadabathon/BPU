// Synchronous first-word-fall-through FIFO with valid/ready on both sides.
// in_ready_o does not depend on out_ready_i (no push-through when full).
module bpu_fifo #(
  parameter int unsigned Width = 32,
  parameter int unsigned Depth = 4
) (
  input  logic                       clk_i,
  input  logic                       rst_ni,

  input  logic                       in_valid_i,
  output logic                       in_ready_o,
  input  logic [Width-1:0]           in_data_i,

  output logic                       out_valid_o,
  input  logic                       out_ready_i,
  output logic [Width-1:0]           out_data_o,

  output logic [$clog2(Depth+1)-1:0] count_o
);

  localparam int unsigned PtrW = (Depth > 1) ? $clog2(Depth) : 1;
  localparam int unsigned CntW = $clog2(Depth + 1);

  // Register storage (packed: never inferred as a RAM). Deep FIFOs should wrap bpu_sram_1r1w.
  logic [Depth-1:0][Width-1:0] mem_q;
  logic [PtrW-1:0]  wptr_q, rptr_q;
  logic [CntW-1:0]  cnt_q;
  logic             push, pop;

  assign in_ready_o  = (cnt_q != CntW'(Depth));
  assign out_valid_o = (cnt_q != '0);
  assign out_data_o  = mem_q[rptr_q];
  assign count_o     = cnt_q;
  assign push        = in_valid_i && in_ready_o;
  assign pop         = out_valid_o && out_ready_i;

  function automatic logic [PtrW-1:0] ptr_incr(input logic [PtrW-1:0] ptr);
    ptr_incr = (ptr == PtrW'(Depth - 1)) ? '0 : ptr + 1'b1;
  endfunction

  always_ff @(posedge clk_i) begin
    if (push) mem_q[wptr_q] <= in_data_i;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wptr_q <= '0;
      rptr_q <= '0;
      cnt_q  <= '0;
    end else begin
      if (push) wptr_q <= ptr_incr(wptr_q);
      if (pop)  rptr_q <= ptr_incr(rptr_q);
      if (push && !pop)      cnt_q <= cnt_q + 1'b1;
      else if (pop && !push) cnt_q <= cnt_q - 1'b1;
    end
  end

endmodule
