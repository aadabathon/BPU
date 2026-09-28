// Leading-zero count. Returns Width when the input is zero.
module bpu_lzc #(
  parameter int unsigned Width = 32
) (
  input  logic [Width-1:0]           in_i,
  output logic [$clog2(Width+1)-1:0] cnt_o
);

  localparam int unsigned CntW = $clog2(Width + 1);

  // The highest set bit wins because it is visited last.
  always_comb begin
    cnt_o = CntW'(Width);
    for (int unsigned i = 0; i < Width; i++) begin
      if (in_i[i]) cnt_o = CntW'(Width - 1 - i);
    end
  end

endmodule
